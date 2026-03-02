import os
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from epitaph.pjsvd import find_optimal_perturbation, apply_correction

os.environ['JAX_PLATFORMS'] = 'cpu,cuda'

# --- 1. ResNet-18 Architecture (Flax NNX) ---

class BasicBlock(nnx.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), 
                              strides=(stride, stride), padding=1, use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(out_channels, rngs=rngs)
        
        self.conv2 = nnx.Conv(out_channels, out_channels, kernel_size=(3, 3), 
                              strides=(1, 1), padding=1, use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(out_channels, rngs=rngs)
        
        self.shortcut = nnx.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nnx.Sequential(
                nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), 
                         strides=(stride, stride), use_bias=False, rngs=rngs),
                nnx.BatchNorm(out_channels, rngs=rngs)
            )
            
    def __call__(self, x):
        out = nnx.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nnx.relu(out)
        return out

class ResNet(nnx.Module):
    def __init__(self, block, num_blocks, num_classes=10, rngs: nnx.Rngs = None):
        self.in_channels = 64
        self.conv1 = nnx.Conv(3, 64, kernel_size=(3, 3), strides=(1, 1), padding=1, use_bias=False, rngs=rngs)
        self.bn1 = nnx.BatchNorm(64, rngs=rngs)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, rngs=rngs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, rngs=rngs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, rngs=rngs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, rngs=rngs)
        
        self.linear = nnx.Linear(512 * block.expansion, num_classes, rngs=rngs)
        
    def _make_layer(self, block, out_channels, num_blocks, stride, rngs: nnx.Rngs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, rngs=rngs))
            self.in_channels = out_channels * block.expansion
        return nnx.Sequential(*layers)
        
    def __call__(self, x):
        """Forward pass. Expects BxHxWxC format."""
        out = nnx.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = jnp.mean(out, axis=(1, 2)) # Global Average Pooling
        out = self.linear(out)
        return out

def ResNet18(num_classes=10, rngs: nnx.Rngs = None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, rngs=rngs)

# --- 2. Data Loading & Training ---

import pickle

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data'], dict[b'labels']

def load_cifar10(data_dir='cifar-10-batches-py'):
    # Load training data
    x_train, y_train = [], []
    for i in range(1, 6):
        data, labels = load_cifar_batch(os.path.join(data_dir, f'data_batch_{i}'))
        x_train.append(data)
        y_train.extend(labels)
    
    x_train = np.concatenate(x_train).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(y_train, dtype=np.int32)
    
    # Load test data
    x_test, labels2 = load_cifar_batch(os.path.join(data_dir, 'test_batch'))
    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(labels2, dtype=np.int32)
    
    # Normalize to [0, 1] then by mean/std
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
    
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    return x_train, y_train, x_test, y_test

def batch_iterator(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, n, batch_size):
        batch_idx = indices[i:i + batch_size]
        yield x[batch_idx], y[batch_idx]

def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    print(f"Training ResNet-18 on CIFAR-10 for {epochs} epochs...")
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    params = nnx.state(model, nnx.Param)
    batch_stats = nnx.state(model, nnx.BatchStat)
    opt_state = optimizer.init(params)
    
    @nnx.jit
    def train_step(model, opt_state, x, y):
        # We need to manually handle train mode for BatchNorm in nnx
        # Setting model flags is possible, or we just rely on nnx defaults
        
        def loss_fn(model):
            logits = model(x)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
            return loss, logits
            
        (loss, logits), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_opt = optimizer.update(nnx.state(grads, nnx.Param), opt_state, nnx.state(model, nnx.Param))
        nnx.update(model, optax.apply_updates(nnx.state(model, nnx.Param), updates))
        return loss, new_opt
        
    @nnx.jit
    def eval_step(model, x, y):
        logits = model(x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=y))
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return loss, acc
        
    for epoch in range(epochs):
        # Train
        model.train() # Set BatchNorm to train mode
        train_loss = 0.0
        num_batches_train = 0
        for batch_idx, (inputs, targets) in enumerate(batch_iterator(x_train, y_train, batch_size)):
            loss, opt_state = train_step(model, opt_state, inputs, targets)
            train_loss += loss.item()
            num_batches_train += 1
            
        # Eval
        model.eval() # Set BatchNorm to eval mode
        test_loss = 0.0
        test_acc = 0.0
        num_batches_test = 0
        for inputs, targets in batch_iterator(x_test, y_test, batch_size, shuffle=False):
            loss, acc = eval_step(model, inputs, targets)
            test_loss += loss.item()
            test_acc += acc.item()
            num_batches_test += 1
            
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/num_batches_train:.4f} | "
              f"Test Loss: {test_loss/num_batches_test:.4f} | "
              f"Test Acc: {test_acc/num_batches_test:.4f}")
              
    return model

# --- 3. PJSVD Integration ---

class ResNetEnsemble:
    def __init__(self, base_model, perturbations):
        self.base_model = base_model
        self.perturbations = perturbations

    def manual_forward(self, x, W_final_new, b_final_new):
        """Forward pass swapping only the very last layer."""
        
        # We must re-implement the forward pass up to the last layer
        out = nnx.relu(self.base_model.bn1(self.base_model.conv1(x)))
        out = self.base_model.layer1(out)
        out = self.base_model.layer2(out)
        out = self.base_model.layer3(out)
        out = self.base_model.layer4(out)
        h = jnp.mean(out, axis=(1, 2))
        
        # Apply the PJSVD modified linear layer
        logits = h @ W_final_new + b_final_new
        return logits

    def predict_proba(self, x):
        ys = []
        for W_new, b_new in self.perturbations:
            logits = self.manual_forward(x, W_new, b_new)
            probs = nnx.softmax(logits, axis=-1)
            ys.append(probs)
        return jnp.stack(ys, axis=0)

def extract_features(model, x_train, y_train, max_batches=10, batch_size=128):
    model.eval()
    all_features = []
    
    @nnx.jit
    def get_h(model, x):
        out = nnx.relu(model.bn1(model.conv1(x)))
        out = model.layer1(out)
        out = model.layer2(out)
        out = model.layer3(out)
        out = model.layer4(out)
        h = jnp.mean(out, axis=(1, 2))
        return h

    for i, (inputs, _) in enumerate(batch_iterator(x_train, y_train, batch_size)):
        if i >= max_batches: break
        h = get_h(model, inputs)
        all_features.append(h)
    return jnp.concatenate(all_features, axis=0)

def run_pjsvd_resnet(model, x_train, y_train, n_directions=40, n_perturbations=100, perturbation_size=10.0):
    print("\n--- Running PJSVD on Final Layer ---")
    
    # We apply PJSVD onto the input of the final Linear layer to perturb its outputs.
    # The linear layer takes `h` (shape 512) to `logits` (shape 10).
    
    print("Extracting feature representations `h` over subset of training data...")
    h_train = extract_features(model, x_train, y_train, max_batches=20) # Use ~2560 samples for nullspace
    
    # We want to perturb the weights of the Linear layer.
    # model.linear shape: kernel (512, 10), bias (10,)
    
    W_orig = model.linear.kernel.get_value()
    b_orig = model.linear.bias.get_value()
    
    def feature_to_logits(W):
        return h_train @ W + b_orig
        
    print(f"Finding {n_directions} Orthogonal Null Space Directions...")
    directions = []
    sigmas = []
    
    for k in range(n_directions):
        orth_constraint = jnp.stack(directions) if len(directions) > 0 else None
            
        v_opt, sigma = find_optimal_perturbation(
            feature_to_logits, 
            W_orig, 
            max_iter=500, 
            orthogonal_directions=orth_constraint
        )
        directions.append(v_opt)
        sigmas.append(sigma)
        print(f"  Direction {k+1}: Residual Sigma = {sigma:.6f}")
        
    v_opts = jnp.stack(directions)
    
    print(f"Generating {n_perturbations} ensemble members (size={perturbation_size})...")
    perturbations = []
    
    n_pairs = n_perturbations // 2
    for i in range(n_pairs):
        z = np.random.normal(0, 1, size=n_directions)
        safe_sigmas = jnp.array(sigmas) + 1e-6
        coeffs = z / safe_sigmas
        
        coeffs = coeffs / np.linalg.norm(coeffs) * perturbation_size
        
        weighted_vs = jnp.reshape(coeffs, (-1, 1, 1)) * v_opts
        total_perturbation = jnp.sum(weighted_vs, axis=0)
        
        W_new_pos = W_orig + total_perturbation
        perturbations.append((W_new_pos, b_orig))
        
        W_new_neg = W_orig - total_perturbation
        perturbations.append((W_new_neg, b_orig))
        
    return ResNetEnsemble(model, perturbations)

def evaluate_uncertainty(model, ensemble, x_test, y_test, batch_size=128, n_batches=5):
    print("\n--- Evaluating Predictive Entropy ---")
    model.eval()
    
    all_clean_entropies = []
    all_noisy_entropies = []
    
    for i, (inputs, targets) in enumerate(batch_iterator(x_test, y_test, batch_size, shuffle=False)):
        if i >= n_batches: break
        
        # 1. Clean Data
        ens_probs_clean = ensemble.predict_proba(inputs)
        mean_probs_clean = jnp.mean(ens_probs_clean, axis=0)
        entropy_clean = -jnp.sum(mean_probs_clean * jnp.log2(mean_probs_clean + 1e-9), axis=-1)
        all_clean_entropies.append(entropy_clean)
        
        # 2. OOD Data (Add massive Gaussian noise)
        noisy_inputs = inputs + np.random.normal(0, 2.0, size=inputs.shape).astype(np.float32)
        ens_probs_noisy = ensemble.predict_proba(noisy_inputs)
        mean_probs_noisy = jnp.mean(ens_probs_noisy, axis=0)
        entropy_noisy = -jnp.sum(mean_probs_noisy * jnp.log2(mean_probs_noisy + 1e-9), axis=-1)
        all_noisy_entropies.append(entropy_noisy)
        
    clean_entropies = np.concatenate(all_clean_entropies)
    noisy_entropies = np.concatenate(all_noisy_entropies)
    
    print(f"Avg Predictive Entropy (Clean CIFAR-10): {np.mean(clean_entropies):.4f} bits")
    print(f"Avg Predictive Entropy (Noisy OOD Data): {np.mean(noisy_entropies):.4f} bits")
    
    return clean_entropies, noisy_entropies

if __name__ == "__main__":
    print("=== EXPERIMENT: CIFAR-10 Classification with ResNet-18 & PJSVD ===")
    
    x_train, y_train, x_test, y_test = load_cifar10()
    
    print("Initializing ResNet-18...")
    rngs = nnx.Rngs(42)
    model = ResNet18(num_classes=10, rngs=rngs)
    
    # Train for a few epochs (enough to get sensible features, normally takes 50-100)
    model = train_model(model, x_train, y_train, x_test, y_test, epochs=5)
    
    # Run PJSVD. Last layer size is 512 -> 10. We can find 50 directions easily.
    ensemble = run_pjsvd_resnet(model, x_train, y_train, n_directions=50, n_perturbations=50, perturbation_size=20.0)
    
    evaluate_uncertainty(model, ensemble, x_test, y_test, n_batches=10)


print("Reading benchmark_output.txt with error ignoring...")
try:
    with open('final_mbrl_log.txt', 'r', encoding='utf-8', errors='ignore') as f:
        print(f.read())
except Exception as e:
    print(f"Failed to read as utf-8: {e}")
    try:
        with open('final_mbrl_log.txt', 'r', encoding='utf-16', errors='ignore') as f:
            print(f.read())
    except Exception as e:
        print(f"Failed to read as utf-16: {e}")

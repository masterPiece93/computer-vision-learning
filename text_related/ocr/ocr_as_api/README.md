# OCR

**Exposing OCR service as `cloud function` api**

## Prerequisite

- please create a `.env` file at root
- please create a virtualenv and install dependencies in it

### Vegeta Tool

Installation

```sh
# 1. Download the latest 64-bit Linux archive
wget https://github.com/tsenart/vegeta/releases/download/v12.12.0/vegeta_12.12.0_linux_amd64.tar.gz

# 2. Extract the archive
tar -xf vegeta_12.12.0_linux_amd64.tar.gz

# 3. Move the binary into your system-wide executable path
sudo mv vegeta /usr/local/bin/

# 4. Clean up the downloaded file
rm vegeta_12.12.0_linux_amd64.tar.gz
```

Check
```sh
vegeta -version
```

1. If you want a worker to "wait for a response" before sending the next one
- By default, Vegeta acts as an open-loop system, meaning it fires requests at a constant **-rate** regardless of how long the server takes to respond. If you want to model a group of concurrent users who wait for a response before sending their next request, combine a rate of zero (`-rate=0` ) with the max workers flag:

    ```bash
    #Simulates 10 concurrent users who wait for their previous request to finish before sending a new one
    vegeta attack -targets=targets.txt -rate=0 -max-workers=10 -duration=30s
    ```

2. If you want a "timeout" (giving up after waiting too long)
- If you want to specify how long Vegeta should wait for a slow server to respond before dropping the connection, use the `-timeout` flag. The default is 30 seconds:

    ```bash
    # If the server doesn't respond within 5 seconds, drop the request ocrand mark it as a failure
    vegeta attack -targets=targets.txt -timeout=5s -duration=10s
    ```

3. If you want to "wait" or pause between requests (lowering the frequency)
- Vegeta doesn't have a "sleep" or "delay" argument. Instead, you control the delay by lowering the `-rate` flag. If you want it to wait a full second or more between hitting the endpoint, pass a fractional rate or a custom time unit:

    ```bash
    # Wait 1 second between each request (1 request per second)
    vegeta attack -targets=targets.txt -rate=1/1s -duration=10s

    # Wait 5 seconds between each request (1 request every 5 seconds)
    vegeta attack -targets=targets.txt -rate=1/5s -duration=30s
    ```

#### Test Runs & Results

Test #1
```bash
echo "GET http://127.0.0.1:8080/?image_url=x.png" | vegeta attack -rate=1/5s -max-workers=1 -duration=30s | tee results.bin | vegeta report
```

Result
```bash
Requests      [total, rate, throughput]         6, 0.24, 0.23
Duration      [total, attack, wait]             26.358s, 24.997s, 1.361s
Latencies     [min, mean, 50, 90, 95, 99, max]  1.349s, 1.38s, 1.375s, 1.41s, 1.41s, 1.41s, 1.41s
Bytes In      [total, mean]                     240, 40.00
Bytes Out     [total, mean]                     0, 0.00
Success       [ratio]                           100.00%
Status Codes  [code:count]                      200:6  
Error Set:
```

- x - x -


Test #2
```bash
echo "GET http://127.0.0.1:8080/?image_url=x.png" | vegeta attack -rate=2 -max-workers=10 -duration=30s | tee results.bin | vegeta report
```

Result
```bash
Requests      [total, rate, throughput]         25, 0.83, 0.73
Duration      [total, attack, wait]             34.078s, 30.21s, 3.868s
Latencies     [min, mean, 50, 90, 95, 99, max]  3.369s, 11.687s, 11.34s, 17.304s, 17.434s, 17.506s, 17.506s
Bytes In      [total, mean]                     1000, 40.00
Bytes Out     [total, mean]                     0, 0.00
Success       [ratio]                           100.00%
Status Codes  [code:count]                      200:25  
Error Set:
```

- x - x -

Test #3
```bash
echo "GET http://127.0.0.1:8080/?image_url=x.png" | vegeta attack -rate=2 -max-workers=10 -duration=10s | tee results.bin | vegeta report
```

Result
```bash
Requests      [total, rate, throughput]         14, 1.36, 0.85
Duration      [total, attack, wait]             16.379s, 10.311s, 6.068s
Latencies     [min, mean, 50, 90, 95, 99, max]  3.795s, 9.003s, 9.229s, 11.675s, 12.106s, 12.23s, 12.23s
Bytes In      [total, mean]                     560, 40.00
Bytes Out     [total, mean]                     0, 0.00
Success       [ratio]                           100.00%
Status Codes  [code:count]                      200:14  
Error Set:
```

Test #4
```bash
echo "GET http://127.0.0.1:8080/?image_url=x.png" | vegeta attack -rate=1 -max-workers=10 -duration=10s | tee results.bin | vegeta report
```

Result
```bash
Requests      [total, rate, throughput]         10, 1.11, 0.91
Duration      [total, attack, wait]             10.931s, 8.999s, 1.932s
Latencies     [min, mean, 50, 90, 95, 99, max]  1.639s, 2.258s, 2.089s, 2.9s, 2.927s, 2.927s, 2.927s
Bytes In      [total, mean]                     400, 40.00
Bytes Out     [total, mean]                     0, 0.00
Success       [ratio]                           100.00%
Status Codes  [code:count]                      200:10  
Error Set:
```

- plotting : `vegeta plot bench_results.bin > benchmark_graph.html`

### Cloud RUn Functions

can we specify python version for deploying cloud functions ?
- yes
    ```txt
    gcloud functions deploy MY_FUNCTION_NAME \
    --runtime=python311 \
    --trigger-http \
    --allow-unauthenticated
    ```
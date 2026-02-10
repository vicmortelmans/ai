export VLLM_LOGGING_LEVEL=DEBUG
# Run in a subshell so `OUTPUT` is set for the command and available for the tee filename
#OUTPUT=output1 ; python3 infer-batch.py --no-prefix-caching --no-continuous-batching --no-speculative-decoding --output-prefix "$OUTPUT" 2>&1 | tee /hfcache/output/${OUTPUT}_script_output.txt
#OUTPUT=output2 ; python3 infer-batch.py --no-prefix-caching --continuous-batching --no-speculative-decoding --output-prefix "$OUTPUT" 2>&1 | tee /hfcache/output/${OUTPUT}_script_output.txt
#OUTPUT=output3 ; python3 infer-batch.py --no-prefix-caching --no-continuous-batching --speculative-decoding --output-prefix "$OUTPUT" 2>&1 | tee /hfcache/output/${OUTPUT}_script_output.txt
#OUTPUT=output4 ; python3 infer-batch.py --no-prefix-caching --continuous-batching --speculative-decoding --output-prefix "$OUTPUT" 2>&1 | tee /hfcache/output/${OUTPUT}_script_output.txt
#OUTPUT=output5 ; python3 infer-batch.py --prefix-caching --no-continuous-batching --no-speculative-decoding --output-prefix "$OUTPUT" 2>&1 | tee /hfcache/output/${OUTPUT}_script_output.txt
#OUTPUT=output6 ; python3 infer-batch.py --prefix-caching --continuous-batching --no-speculative-decoding --output-prefix "$OUTPUT" 2>&1 | tee /hfcache/output/${OUTPUT}_script_output.txt
#OUTPUT=output7 ; python3 infer-batch.py --prefix-caching --no-continuous-batching --speculative-decoding --output-prefix "$OUTPUT" 2>&1 | tee /hfcache/output/${OUTPUT}_script_output.txt
OUTPUT=output8 ; python3 infer-batch.py --prefix-caching --continuous-batching --speculative-decoding --output-prefix "$OUTPUT" 2>&1 | tee /hfcache/output/${OUTPUT}_script_output.txt
export VLLM_LOGGING_LEVEL=DEBUG

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <test_directory_path>"
    exit 1
fi

TEST_DIR=$1
BASE_DIR=$(dirname "${TEST_DIR}")

INPUT_DIR="${BASE_DIR}/annotated"
PROMPTS_PATH="${TEST_DIR}/prompts"
REFERENCE_DIR="${BASE_DIR}/reference"
CSV_FILE="${BASE_DIR}.csv" # Following the literal instruction: "$BASE_DIR.csv"

# Extract TEST_NAME for CSV logging
TEST_NAME=$(basename "${TEST_DIR}")

# Ensure the CSV header is written only once.
# Header fields: TestName,OutputSubdir,PromptBasename,DiffCount
if [ ! -f "$CSV_FILE" ]; then
    echo "Timestamp,TestName,OutputSubdir,PromptBasename,DiffCount" > "$CSV_FILE"
fi

for PROMPT_FILE in "${PROMPTS_PATH}"/*.txt; do
    # Handle cases where no .txt files are found
    [ -e "$PROMPT_FILE" ] || continue
    
    # Extract filename without extension (e.g., 'base' from 'base.txt')
    PROMPT_BASENAME_NO_EXT=$(basename "$PROMPT_FILE" .txt)
    
    OUTPUT_DIR="${TEST_DIR}/output/${PROMPT_BASENAME_NO_EXT}"
    META_DIR="${OUTPUT_DIR}/meta"
    DIFFERENCES_DIR="${META_DIR}/differences"
    
    mkdir -p "$META_DIR"
    mkdir -p "$DIFFERENCES_DIR"
    
    python3 "${SCRIPT_DIR}/infer-batch.py" --prefix-caching --continuous-batching --speculative-decoding --input-prefix "$INPUT_DIR" --prompt-file "$PROMPT_FILE" --output-prefix "$OUTPUT_DIR" --meta-prefix "$META_DIR" --dry-run 2>&1 | tee "${META_DIR}/script_output.txt"

    # --- Start of new loop for wdiff and CSV logging ---
    # Iterate over all newly created txt files in $OUTPUT_DIR
    for GENERATED_OUTPUT_FILE in "${OUTPUT_DIR}"/*.txt; do
        [ -e "$GENERATED_OUTPUT_FILE" ] || continue # Skip if no .txt files found

        GENERATED_BASENAME=$(basename "$GENERATED_OUTPUT_FILE") # e.g., base.txt
        GENERATED_BASENAME_NO_EXT="${GENERATED_BASENAME%.txt}" # e.g., base

        REFERENCE_FILE="${REFERENCE_DIR}/${GENERATED_BASENAME}"
        DIFF_OUTPUT_FILE="${DIFFERENCES_DIR}/${GENERATED_BASENAME_NO_EXT}-differences.txt"

        if [ ! -f "$REFERENCE_FILE" ]; then
            echo "Warning: Reference file not found for comparison: $REFERENCE_FILE" | tee -a "${META_DIR}/script_output.txt"
            DIFF_LINE_COUNT="N/A" # Indicate no comparison was made
            touch "$DIFF_OUTPUT_FILE" # Create an empty diff file
        else
            # Run wdiff, capture output, count lines, and save to file
            TEMP_DIFF_OUTPUT=$(mktemp)
            wdiff --no-common "$GENERATED_OUTPUT_FILE" "$REFERENCE_FILE" | grep -v '==' > "$TEMP_DIFF_OUTPUT"
            DIFF_LINE_COUNT=$(wc -l < "$TEMP_DIFF_OUTPUT")
            mv "$TEMP_DIFF_OUTPUT" "$DIFF_OUTPUT_FILE"
            echo "Generated diff for $GENERATED_BASENAME to $DIFF_OUTPUT_FILE (Lines: $DIFF_LINE_COUNT)" | tee -a "${META_DIR}/script_output.txt"
        fi
        
        # Append to CSV: TestName,OutputSubdir,PromptBasename,DiffCount
        # Assuming the generated output file name matches the prompt basename.
        TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
        CSV_LINE="${TIMESTAMP},${TEST_NAME},output,${PROMPT_BASENAME_NO_EXT},${DIFF_LINE_COUNT}"
        echo "$CSV_LINE" >> "$CSV_FILE"
    done
    # --- End of new loop ---

done

# If a 'base' prompt exists, compare other prompts' differences to the base's differences
BASE_DIFF_DIR="${TEST_DIR}/output/base/meta/differences"
if [ -d "$BASE_DIFF_DIR" ]; then
    for OTHER_PROMPT_FILE in "${PROMPTS_PATH}"/*.txt; do
        [ -e "$OTHER_PROMPT_FILE" ] || continue
        OTHER_BASENAME=$(basename "$OTHER_PROMPT_FILE" .txt)
        if [ "$OTHER_BASENAME" != "base" ]; then
            OTHER_META_DIR="${TEST_DIR}/output/${OTHER_BASENAME}/meta"
            if [ -d "${OTHER_META_DIR}/differences" ]; then
                diff -r "$BASE_DIFF_DIR" "${OTHER_META_DIR}/differences" > "${OTHER_META_DIR}/diff_against_base.txt"
            fi
        fi
    done
fi

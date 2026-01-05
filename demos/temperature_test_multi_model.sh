#!/bin/bash
# Enhanced Temperature Test - Compare multiple models with different temperatures
# Shows how temperature affects response quality across different model families

# Source the wrapper URL configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/get_wrapper_url.sh"
HOST="$WRAPPER_URL"
PROMPT="Explain what a binary search algorithm does in one sentence."

# Colors for better output (optional)
BOLD='\033[1m'
RESET='\033[0m'

# Function to format duration in hours:minutes:seconds
format_duration() {
    local seconds=$1
    if (( $(echo "$seconds < 1" | bc -l) )); then
        # Less than 1 second: show milliseconds
        echo "$(echo "$seconds * 1000" | bc -l | xargs printf "%.0f")ms"
    elif (( $(echo "$seconds < 60" | bc -l) )); then
        # Less than 1 minute: show seconds with 2 decimals
        echo "$(printf "%.2f" $seconds)s"
    elif (( $(echo "$seconds < 3600" | bc -l) )); then
        # Less than 1 hour: show minutes:seconds
        local minutes=$(echo "$seconds / 60" | bc)
        local secs=$(echo "$seconds % 60" | bc)
        echo "${minutes}m ${secs}s"
    else
        # 1 hour or more: show hours:minutes:seconds
        local hours=$(echo "$seconds / 3600" | bc)
        local minutes=$(echo "($seconds % 3600) / 60" | bc)
        local secs=$(echo "$seconds % 60" | bc)
        echo "${hours}h ${minutes}m ${secs}s"
    fi
}

# Function to run test and extract metrics
run_test() {
    local model=$1
    local temp=$2
    local desc=$3
    local temp_param=""

    if [ -n "$temp" ]; then
        temp_param="\"temperature\": $temp,"
    fi

    local start=$(date +%s.%N)
    local response=$(curl -s -X POST "$HOST/chat" \
      -H "Content-Type: application/json" \
      -d "{
        \"message\": \"$PROMPT\",
        \"model\": \"$model\",
        \"mcp_server\": \"\",
        $temp_param
        \"dummy\": null
      }" | sed 's/, *"dummy": *null//')
    local end=$(date +%s.%N)
    local elapsed=$(echo "$end - $start" | bc -l)

    echo "$response|$elapsed|$desc|$model"
}

# Get available models from API
get_models() {
    curl -s "$HOST/models" | jq -r '.models[].name'
}

# Function to print table row
print_row() {
    local result=$1
    local temp=$2

    local response=$(echo "$result" | cut -d'|' -f1)
    local elapsed=$(echo "$result" | cut -d'|' -f2)
    local desc=$(echo "$result" | cut -d'|' -f3)

    local tps=$(echo "$response" | jq -r '.metrics.tokens_per_second // 0')
    local tokens=$(echo "$response" | jq -r '.metrics.completion_tokens // 0')
    local total_dur=$(echo "$response" | jq -r '.metrics.total_duration_s // 0')
    local resp_text=$(echo "$response" | jq -r '.response')
    local resp_preview="${resp_text:0:37}"
    if [ ${#resp_text} -gt 40 ]; then
        resp_preview="${resp_preview}..."
    fi

    local elapsed_fmt=$(format_duration $elapsed)
    local total_fmt=$(format_duration $total_dur)

    printf "%-20s %-10.2f %-10s %-12s %-15s %-25s %-40s\n" "$temp" "$tps" "$tokens" "$elapsed_fmt" "$total_fmt" "$desc" "$resp_preview"
}

echo "========================================================================================================================================"
echo "ENHANCED TEMPERATURE TEST - MULTI-MODEL COMPARISON"
echo "========================================================================================================================================"

# Get available models
echo ""
echo "Fetching available models from Ollama..."
mapfile -t available_models < <(get_models)

if [ ${#available_models[@]} -eq 0 ]; then
    echo "No models available. Please ensure Ollama is running and has models installed."
    exit 1
fi

echo ""
echo "Available models:"
for i in "${!available_models[@]}"; do
    echo "  $((i+1)). ${available_models[$i]}"
done

echo ""
echo "Enter model numbers to test (comma-separated, or 'all' for all models):"
echo "Example: 1,3,5  or  all"
read -p "> " user_input

# Parse user input
IFS=',' read -ra selections <<< "$user_input"
selected_models=()

if [[ "$user_input" == "all" ]]; then
    selected_models=("${available_models[@]}")
else
    for selection in "${selections[@]}"; do
        selection=$(echo "$selection" | xargs) # trim whitespace
        idx=$((selection - 1))
        if [ $idx -ge 0 ] && [ $idx -lt ${#available_models[@]} ]; then
            selected_models+=("${available_models[$idx]}")
        fi
    done
fi

if [ ${#selected_models[@]} -eq 0 ]; then
    echo "No valid models selected."
    exit 1
fi

echo ""
echo "Testing ${#selected_models[@]} model(s) with different temperatures"
echo "Prompt: '$PROMPT'"
echo ""

# Temperature configurations
temps=(0.1 "" 0.8 1.5)
descs=("Very Low (Deterministic)" "Default from Config" "Medium (Balanced)" "High (Creative)")

# Store results
declare -A results

total_tests=$((${#selected_models[@]} * ${#temps[@]}))
current_test=0

# Run all tests
for model in "${selected_models[@]}"; do
    for i in "${!temps[@]}"; do
        temp="${temps[$i]}"
        desc="${descs[$i]}"
        current_test=$((current_test + 1))

        echo "[$current_test/$total_tests] Testing $model with $desc (temp=$temp)..."

        result=$(run_test "$model" "$temp" "$desc")
        results["${model}|${temp}"]="$result"
    done
done

# Display results by model
echo ""
echo "========================================================================================================================================"
echo "RESULTS BY MODEL"
echo "========================================================================================================================================"

for model in "${selected_models[@]}"; do
    echo ""
    echo "Model: $model"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"

    # Table header
    printf "%-20s %-10s %-10s %-12s %-15s %-25s %-40s\n" "Temperature" "TPS" "Tokens" "Time" "Total Duration" "Description" "Response Preview"
    echo "----------------------------------------------------------------------------------------------------------------------------------------"

    for i in "${!temps[@]}"; do
        temp="${temps[$i]}"
        temp_display="${temp:-default (0.2)}"
        result="${results[${model}|${temp}]}"

        if [ -n "$result" ]; then
            print_row "$result" "$temp_display"
        fi
    done
done

# Cross-model comparison if multiple models
if [ ${#selected_models[@]} -gt 1 ]; then
    echo ""
    echo "========================================================================================================================================"
    echo "CROSS-MODEL COMPARISON (at each temperature)"
    echo "========================================================================================================================================"

    for i in "${!temps[@]}"; do
        temp="${temps[$i]}"
        desc="${descs[$i]}"
        temp_display="${temp:-default (0.2)}"

        echo ""
        echo "Temperature: $temp_display - $desc"
        echo "----------------------------------------------------------------------------------------------------------------------------------------"

        # Table header
        printf "%-35s %-10s %-10s %-12s %-15s %-50s\n" "Model" "TPS" "Tokens" "Time" "Total Duration" "Response Preview"
        echo "----------------------------------------------------------------------------------------------------------------------------------------"

        for model in "${selected_models[@]}"; do
            result="${results[${model}|${temp}]}"

            if [ -n "$result" ]; then
                response=$(echo "$result" | cut -d'|' -f1)
                elapsed=$(echo "$result" | cut -d'|' -f2)

                tps=$(echo "$response" | jq -r '.metrics.tokens_per_second // 0')
                tokens=$(echo "$response" | jq -r '.metrics.completion_tokens // 0')
                total_dur=$(echo "$response" | jq -r '.metrics.total_duration_s // 0')
                resp_text=$(echo "$response" | jq -r '.response')
                resp_preview="${resp_text:0:47}"
                if [ ${#resp_text} -gt 50 ]; then
                    resp_preview="${resp_preview}..."
                fi

                elapsed_fmt=$(format_duration $elapsed)
                total_fmt=$(format_duration $total_dur)

                printf "%-35s %-10.2f %-10s %-12s %-15s %-50s\n" "$model" "$tps" "$tokens" "$elapsed_fmt" "$total_fmt" "$resp_preview"
            fi
        done
    done
fi

# Full responses
echo ""
echo "========================================================================================================================================"
echo "FULL RESPONSES"
echo "========================================================================================================================================"

for model in "${selected_models[@]}"; do
    echo ""
    echo "========================================================================================================================================"
    echo "MODEL: $model"
    echo "========================================================================================================================================"

    for i in "${!temps[@]}"; do
        temp="${temps[$i]}"
        desc="${descs[$i]}"
        temp_display="${temp:-default (0.2)}"
        result="${results[${model}|${temp}]}"

        if [ -n "$result" ]; then
            response=$(echo "$result" | cut -d'|' -f1)
            resp_text=$(echo "$response" | jq -r '.response')

            echo ""
            echo "$((i+1)). $desc (temp=$temp_display):"
            echo "   $resp_text"
        fi
    done
done

# Summary
echo ""
echo "========================================================================================================================================"
echo "SUMMARY & INSIGHTS"
echo "========================================================================================================================================"
echo ""
echo "Temperature Guidelines:"
echo "  • Lower temperature (0.1-0.3): Best for factual tasks, coding, math"
echo "  • Medium temperature (0.7-1.0): Good for natural conversation"
echo "  • Higher temperature (1.5+): Best for creative writing, brainstorming"
echo ""
echo "Metrics Explanation:"
echo "  • TPS = Tokens Per Second (generation speed)"
echo "  • Time = Total request time including network overhead"
echo "  • Total Duration = Ollama processing time"

if [ ${#selected_models[@]} -gt 1 ]; then
    echo ""
    echo "Model Comparison Tips:"
    echo "  • Smaller models (1-3B) are faster but may be less accurate"
    echo "  • Larger models (7B+) are slower but typically more capable"
    echo "  • Quantization level (Q4, Q5, etc.) affects both speed and quality"
fi

echo ""
echo "Demo complete!"

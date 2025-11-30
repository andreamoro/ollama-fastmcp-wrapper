#!/bin/bash
# Temperature Test Demo - Compare different temperature settings
# Shows how temperature affects response consistency and creativity

HOST="http://localhost:8000"
PROMPT="Explain what a binary search algorithm does in one sentence."

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
    local temp=$1
    local desc=$2
    local temp_param=""

    if [ -n "$temp" ]; then
        temp_param="\"temperature\": $temp,"
    fi

    local start=$(date +%s.%N)
    local response=$(curl -s -X POST "$HOST/chat" \
      -H "Content-Type: application/json" \
      -d "{
        \"message\": \"$PROMPT\",
        \"model\": \"llama3.2:3b\",
        \"mcp_server\": \"\",
        $temp_param
        \"dummy\": null
      }" | sed 's/, *"dummy": *null//')
    local end=$(date +%s.%N)
    local elapsed=$(echo "$end - $start" | bc -l)

    echo "$response|$elapsed|$desc"
}

echo "=== Temperature Test Demo ==="
echo "Testing the same prompt with different temperature values"
echo "Prompt: '$PROMPT'"
echo

# Run all tests
echo "Running test: Very Low (Deterministic) (temp=0.1)..."
result1=$(run_test "0.1" "Very Low (Deterministic)")

echo "Running test: Default from Config (temp=default)..."
result2=$(run_test "" "Default from Config")

echo "Running test: Medium (Balanced) (temp=0.8)..."
result3=$(run_test "0.8" "Medium (Balanced)")

echo "Running test: High (Creative) (temp=1.5)..."
result4=$(run_test "1.5" "High (Creative)")

echo
echo "========================================================================================================================"
echo "RESULTS TABLE"
echo "========================================================================================================================"

# Table header
printf "%-20s %-10s %-10s %-12s %-15s %-25s\n" "Temperature" "TPS" "Tokens" "Time" "Total Duration" "Description"
echo "------------------------------------------------------------------------------------------------------------------------"

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

    local elapsed_fmt=$(format_duration $elapsed)
    local total_fmt=$(format_duration $total_dur)

    printf "%-20s %-10.2f %-10s %-12s %-15s %-25s\n" "$temp" "$tps" "$tokens" "$elapsed_fmt" "$total_fmt" "$desc"
}

print_row "$result1" "0.1"
print_row "$result2" "default (0.2)"
print_row "$result3" "0.8"
print_row "$result4" "1.5"

echo "========================================================================================================================"
echo

# Show responses
echo "RESPONSES:"
echo "------------------------------------------------------------------------------------------------------------------------"

echo
echo "1. Very Low (Deterministic) (temp=0.1):"
echo "   $(echo "$result1" | cut -d'|' -f1 | jq -r '.response')"

echo
echo "2. Default from Config (temp=default (0.2)):"
echo "   $(echo "$result2" | cut -d'|' -f1 | jq -r '.response')"

echo
echo "3. Medium (Balanced) (temp=0.8):"
echo "   $(echo "$result3" | cut -d'|' -f1 | jq -r '.response')"

echo
echo "4. High (Creative) (temp=1.5):"
echo "   $(echo "$result4" | cut -d'|' -f1 | jq -r '.response')"

echo
echo "========================================================================================================================"
echo "SUMMARY"
echo "========================================================================================================================"
echo "Lower temperature (0.1-0.3): Best for factual tasks, coding, math"
echo "Medium temperature (0.7-1.0): Good for natural conversation"
echo "Higher temperature (1.5+): Best for creative writing, brainstorming"
echo
echo "TPS = Tokens Per Second (generation speed)"
echo "Time = Total request time including network overhead"
echo "Total Duration = Ollama processing time"
echo
echo "Demo complete!"

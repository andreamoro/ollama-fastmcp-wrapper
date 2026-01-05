#!/bin/bash
# Utility script to read wrapper host and port from wrapper_config.toml
# This can be sourced by other demo scripts to get the correct wrapper URL
#
# Usage:
#   source demos/get_wrapper_url.sh
#   echo "Wrapper URL: $WRAPPER_URL"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../wrapper_config.toml"

# Default values
WRAPPER_HOST="localhost"
WRAPPER_PORT="8000"

# Read config file if it exists
if [[ -f "$CONFIG_FILE" ]]; then
    # Track which section we're in
    CURRENT_SECTION=""

    while IFS= read -r line || [[ -n "$line" ]]; do
        # Remove leading/trailing whitespace and comments
        line=$(echo "$line" | sed 's/#.*//' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')

        # Skip empty lines
        [[ -z "$line" ]] && continue

        # Check for section headers
        if [[ "$line" =~ ^\[([^]]+)\]$ ]]; then
            CURRENT_SECTION="${BASH_REMATCH[1]}"
            continue
        fi

        # Only process lines in [wrapper] section
        [[ "$CURRENT_SECTION" != "wrapper" ]] && continue

        # Parse host and port
        if [[ "$line" =~ ^host[[:space:]]*=[[:space:]]*[\"']?([^\"']+)[\"']? ]]; then
            value="${BASH_REMATCH[1]}"
            # Convert 0.0.0.0 to localhost for client connections
            if [[ "$value" == "0.0.0.0" ]]; then
                WRAPPER_HOST="localhost"
            else
                WRAPPER_HOST="$value"
            fi
        elif [[ "$line" =~ ^port[[:space:]]*=[[:space:]]*([0-9]+) ]]; then
            WRAPPER_PORT="${BASH_REMATCH[1]}"
        fi
    done < "$CONFIG_FILE"
fi

# Export the wrapper URL for use by calling scripts
export WRAPPER_URL="http://${WRAPPER_HOST}:${WRAPPER_PORT}"

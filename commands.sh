# We have three variants of running code:
# * Debugging: `python -m debugpy --listen localhost:5678 --wait-for-client src/main.py`
# * Interactive non-debugger: `python -i src/main.py`
# * Non-interactive non-debugger: `python src/main.py`

# source project with `hatch shell`.

# Which we will invocate as:
# * `bash ./commands.sh debugging`
# * `bash ./commands.sh interactive`
# * `bash ./commands.sh production`

# Extract command.

case "$1" in 
    debugging) 
        python -m debugpy --listen localhost:5678 --wait-for-client src/main.py
        ;;
    interactive)
        python -i src/main.py
        ;;
    production)
        python src/main.py
        ;;
esac

"""Run the local classifier web app on port 8000."""

from app import app


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)

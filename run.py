from emotune import create_app  # Import create_app() from __init__.py

app = create_app()  # Initialize Flask app

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)  # Start Flask server


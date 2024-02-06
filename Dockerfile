# This sets up the container with Python 3.10 installed.
FROM python:3.10-slim

# This sets the /app directory as the working directory for any RUN, CMD, ENTRYPOINT, or COPY instructions that follow.
WORKDIR /phishing

# This copies everything in your current directory to the /app directory in the container.
COPY . /phishing


# This runs pip install for all the packages listed in your requirements.txt file.
RUN pip install --no-cache-dir -r requirements.txt

# This tells Docker to listen on port 80 at runtime. Port 80 is the standard port for HTTP.
#EXPOSE 80

# This copies your Streamlit configuration file into the /app directory.
#COPY config.toml /phishing/config.toml

# This copies your Streamlit credentials file into the /app directory.
#COPY credentials.toml /app/credentials.toml

# This sets the default command for the container to run the app with Streamlit.
CMD ["streamlit", "run", "application.py"]
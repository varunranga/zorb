# base container
FROM python:3.8

# set the working directory in the container
WORKDIR /src

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip3.8 install -r requirements.txt

# copy source files to /home
COPY src/ .

# give permission to run zorb from command-line
RUN chmod 755 ./zorb.sh

ENTRYPOINT [ "./zorb.sh" ]

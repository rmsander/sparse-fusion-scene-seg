# Bash script to ssh into AWS instance

# Set directory so we have access to .pem file
cd path/to/<ami_key>.pem

# Change permissions on .pem file for ssh capabilities
chmod 0400 <ami_key>.pem

# If we want to use Jupyter notebook on EC2, map local 9999 port to EC2 jupyter 8888 port: First, kill active processes on port
fuser -k 9999/tcp

# Now set port 9999 to listen to port 8888 on AWS machine
ssh -i <ami_key>.pem -NfL 9999:localhost:8888 <remote_user>@<remote_host>


# ssh into AWS instance
ssh -i <ami_key>.pem <remote_user>@<remote_host>





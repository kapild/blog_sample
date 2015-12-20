#!/bin/bash
command_type=$1
host=$2
file=$3

if [ $# != 3 ]; then
	echo "Minimum args is 3"
	echo "exit(1)"
	exit
fi

echo "command_type:"$command_type
echo "host:"$host
echo $file

command_return=""
_get_command() {
	command_type=$1
	if [ $command_type = 'abc' ]; then
		command_return="abc"
	else 
		if [ $command_type = 'xyz' ]; then
			command_return="xyz --url"
		else 
			echo "Not implemented command : "
			exit
		fi
	fi
}

command_return=""
_get_host() {
	host_type=$1
	if [ $host_type = 'local' ]; then
		command_return="localhost"
	else 
		if [ $host_type = 'remote' ]; then
			command_return="some.remote.host"
		else 
			echo "Not implemented host"
			echo $host_type
			exit
		fi
	fi
}

command_return=""
_get_handler() {
	command_type=$1
	if [ $command_type == 'abc' ]; then
		command_return="xyz"
	else 
		if [ $command_type == 'xyz' ]; then
			command_return="endpoint"
		else 
			echo "Not implemented handler : "
			echo $command_type
			exit 1
		fi
	fi
}

command_return=""
_get_post_URL() {
	command_type=$1
	if [ $command_type == 'curl' ]; then
		command_return="-H \"Content-Type: application/pc-json-protobuf\"  "
	else 
		if [ $command_type == 'abc' ]; then
			command_return=""
		else 
			echo "Not implemented handler : "
			echo $command_type
			exit 1
		fi
	fi
}

command_return=""
_get_file() {
	command_type=$1 
	file=$2
	if [ $command_type == 'curl' ]; then
		command_return=' --data-binary @'$file
	else 
		if [ $command_type == 'ag' ]; then
			command_return='--read-request-json '$file
		else 
			echo "Not implemented handler : "
			echo $command_type
			exit 1
		fi
	fi
}

_get_locale() {
	command_type=$1 
	if [ $command_type == 'curl' ]; then
		command_return='-H "extra arg"'
	else 
		if [ $command_type == 'ag' ]; then
			command_return='--locale extra agr'
		else 
			echo "Not implemented handler : "
			echo $command_type
			exit 1
		fi
	fi
}

URL=""
_get_command $command_type 
URL+=$command_return

_get_host $host

URL+=' http://'$command_return

_get_handler $command_type

URL_EN=$URL'/'$command_return'?abc=123'
URL_J=$URL'/'$command_return'?xyz=123'
URL_S=$URL'/'$command_return'?nnn=uuuu'

_get_post_URL $command_type

URL_EN+=' '$command_return
URL_J+=' '$command_return
URL_S+=' '$command_return

_get_locale $command_type

URL_EN+=' '$command_return
URL_J+=' '$command_return
URL_S+=' '$command_return

_get_file $command_type $file

URL_EN+=' '$command_return
URL_J+=' '$command_return
URL_S+=' '$command_return
echo 
echo 

echo "EN:"
echo "    "$URL_E
echo 
echo "Japan:"
echo "    "$URL_J
echo 
echo "Siri:"
echo "    "$URL_S
echo 







#!/usr/bin/env python
import argparse
import os
import sys
import subprocess
import csv

__all__ = []
__version__ = 0.1
__date__ = "2016-11-17"
__updated__ = "2016-11-17"


# memory and hd is GB
class Server:
    def __init__(self, name, memory=None, cpu=None, hd=None):
        self.name = name
        self.memory = memory
        self.cpu = cpu
        self.hd = hd

    def __repr__(self):
        return " repr: Server {0} has {1} GB memory, {2} CPU, and {3} GB hard drive".format(self.name, self.memory, self.cpu, self.hd)

    def __str__(self):
        return "str: Server {0} has {1} GB memory, {2} CPU, and {3} GB hard drive".format(self.name, self.memory, self.cpu, self.hd)


def main():
    try:
        program_name = os.path.basename(sys.argv[0])
        program_version = "v%s" % __version__
        program_build_date = str(__updated__)
        program_version_message = "%%(prog)s %s (%s)" % (program_version, program_build_date)
        program_shortdesc = "python program to get server resources"
        program_license = '''%s
Created by Shawn Xue on %s.
Copyright 2016 Apptio Inc. All rights reserved.
USAGE
python <path to server_resource.py> -p <root password> -f <full path of server list file>
''' % (program_shortdesc, str(__date__))

        parser = argparse.ArgumentParser(description=program_license, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("-p", "--password", dest="root_password", required=True, help="The root password of all servers")
        parser.add_argument("-f", "--file", dest="file", required=True, help="The file including all servers")
        parser.add_argument("-v", "--verbose", dest="verbose", action="count", help="set verbosity level [default: %(default)s]")
        parser.add_argument("-V", "--version", action="version", version=program_version_message)
        args = parser.parse_args()
        result = getServerResources(args)
        print sorted(result, key=lambda server: server.memory, reverse=True)
        writeToFile(result)
        return 0
    except Exception as e:
        indent = " " * len(program_name)
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use " + sys.argv[0] + " --help")
        return 2


def getServerResources(args):
    result = []
    if not os.path.isfile(args.file):
        return result
    with open(args.file, "r") as f:
        for line in f.readlines():
            name = line.strip()
            mem, cpu, hd = getServerComputing(args.root_password, name)
            server = Server(name, mem, cpu, hd)
            result.append(server)
    return result


def getServerComputing(root_password, name):
    cmd = ["sshpass", "-p", root_password, "ssh", "-n", "-o", "StrictHostKeyChecking=no", "root@" + name, "-t"]
    cmd_mem = list(cmd)
    cmd_mem.extend(["free", "-g", "|", "grep", "Mem", "|", "awk", "'{print $2}'"])
    p_mem = runCommand(cmd_mem)
    # calling communicate returns a tuple
    mem = p_mem.communicate()[0].strip()
    cmd_cpu = list(cmd)
    cmd_cpu.append("nproc")
    p_cpu = runCommand(cmd_cpu)
    cpu = p_cpu.communicate()[0].strip()
    hd = 0
    return mem, cpu, hd


def runCommand(cmd):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.wait()
    return proc


def writeToFile(result):
    with open("result.csv", "wb") as f:
        fieldnames = ["name", "memory", "CPU", "HD"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for server in result:
            writer.writerow({"name": server.name, "memory": server.memory, "CPU": server.cpu, "HD": server.hd})

# entry of this program
if __name__ == "__main__":
    sys.exit(main())
import json
import requests
from urllib import quote_plus
from influxdb import InfluxDBClient
from datetime import datetime
from dateutil import parser
from requests.auth import HTTPBasicAuth
from requests.packages.urllib3.exceptions import InsecureRequestWarning
import consul

class Consul_Driver:
    def __init__ ( self ):
        self.token = "haha"
        self.server = "127.0.0.1"
        self.port = 32781

    def get_last_build(self, KeyName):
        consulclient = consul.Consul(host=self.server,port=self.port,scheme='http')
        value = consulclient.kv.get(KeyName)
        if value == "":
            consulclient.kv.put(KeyName,"0")
            value = "0"
        return value

    def set_last_build ( self , KeyName, value ):
        consulclient = consul.Consul ( host=self.server , port=self.port , scheme='http' )
        if value == "":
            consulclient.kv.put ( KeyName , "0" )
        else:
            consulclient.kv.put ( KeyName , value )
        return value
class Vault_Driver:
    pass
class MetricsDB:
    def __init__ ( self , server , port , username , password , database ):
        self.influxdbserver = server
        self.influxdbport = port
        self.influxdbusername = username
        self.influxdbpassword = password
        self.influxdbdatabase = database
    def insert_datapoint(self, data):
        dbclient = InfluxDBClient ( self.influxdbserver , self.influxdbport , self.influxdbusername ,
                                    self.influxdbpassword , self.influxdbdatabase )
        dbclient.write_points ( data )
        return 0
class Measurement ( ):
    def __init__ ( self , measurement , tags , time , fields ):
        self.timeseriesname = measurement
        self.timeseriestags = tags
        self.timeseriestime = time
        self.timeseriesfields = fields
    def get_data ( self ):
        json_body = [{'measurement': '' , 'tags': {} , 'time': '' , 'fields': {}}]
        json_body[ 0 ][ "measurement" ] = self.timeseriesname
        json_body[ 0 ][ "tags" ] = self.timeseriestags
        json_body[ 0 ][ "time" ] = self.timeseriestime
        json_body[ 0 ][ "fields" ] = self.timeseriesfields
        return json_body
class Token:
    def __init__ ( self , productname , role ):
        self.source = productname
        self.role = role

    def get_token ( self ):
        response_cred = {}
        if self.source == "gitlab" and self.role == "ci_metrics":
            self.user = "pass"
            self. token = "haha"
        elif self.source == "teamcity" and self.role == "ci_metrics":
            self.user = "metrics"
            self.token = "haha"
        else:
            raise ValueError("Invalid productname or role")
        response_cred[self.user] = self.token
        return response_cred
class Metrics_Request:
    def __init__(self, url, headers, auth ="", querystring="" ):
        if not url:
            raise ValueError("Missing URL")
        if not headers:
            ValueError("Missing headers")
        self.request_url = url
        self.request_headers = headers
        self.auth = auth
        self.parms = querystring
    def get_json(self):
        ####
        requests.packages.urllib3.disable_warnings ( InsecureRequestWarning )
        api_request = requests.get(self.request_url ,auth=self.auth, headers=self.request_headers , verify=False, params=self.parms )
        if api_request.status_code == 200:
            return api_request.json()
        else:
            return False
class CiCIMetric_SCM_GitLab:
    def __init__(self, host, token,https=True):
        self.api_url = "/api/v3"
        self.headers = {'Accept': 'application/json' , 'Content-type': 'application/json'}
        if not token:
            raise ValueError("missing token value")
        else:
            self.token = token
            self.headers["PRIVATE-TOKEN"] = self.token
        if not host:
            raise ValueError("missing host name")
        if https == True:
            self.host = "https://" + host
        else:
            self.host = "http://" + host

    def get_projects_all(self):
        self.request_url = self.host + self.api_url + "/projects/all"
        data = Metrics_Request ( self.request_url , self.headers )
        dataset = data.get_json()
        resultset = {}
        for project in dataset:
            resultset[project["id"]] = project["path_with_namespace"]
        return resultset

    def get_get_branches(self, projectid):
        self.request_url = self.host + self.api_url + "/projects/" + str(projectid) + "/repository/branches"
        data = Metrics_Request ( self.request_url , self.headers )
        dataset = data.get_json()
        resultset = []
        for branch in dataset:
            resultset.append(branch["name"])
        return resultset

    def write_commit_by_repository (self, projectid , relative_path ):
        self.request_url = self.host + self.api_url + "/projects/" + str ( projectid ) + "/repository/commits"
        data = Metrics_Request ( self.request_url , self.headers )
        dataset = data.get_json()
        for commit in dataset:
            tags = {}
            fields = {}
            time = parser.parse ( commit[ "created_at" ] ).strftime ( "%Y%m%dT%H%M%S+0000" )
            tags[ "ProjectID" ] = projectid
            tags[ "relative_path" ] = relative_path
            tags[ "author_email" ] = commit[ "author_email" ]
            tags[ "short_id" ] = commit[ "short_id" ]
            fields[ "ProjectID" ] = projectid
            fields[ "relative_path" ] = relative_path
            fields[ "author_email" ] = commit[ "author_email" ]
            fields[ "short_id" ] = commit[ "short_id" ]
            datapoint_json = Measurement ( "Ci_CI_GitLab_Commits" , tags , time , fields )
            measurement_json = datapoint_json.get_data ( )
            dbclient = MetricsDB ( "localhost" , "8086" , "admin" , "admin" , "CiCIMetrics" )
            dbclient.insert_datapoint ( measurement_json )
class CiCIMetric_Build_TeamCity:
    def __init__(self, host, username, token,https=True ):
        self.api_url = "/httpAuth/app/rest"
        self.headers = {'Accept': 'application/json' , 'Content-type': 'application/json'}
        self.username = username
        self.token = token
        self.auth = HTTPBasicAuth(self.username, self.token)
        if https == True:
            self.host = "https://" + host
        else:
            self.host = "http://" + host

    def request_problemOccurrences(self, buildID):
        self.request_url = self.host + self.api_url + "/problemOccurrences"
        self.querystring = {"locator": "build:(0)"}
        self.querystring[ "locator" ] = "build:(" + str ( buildID ) + ")"
        data = Metrics_Request ( self.request_url , self.headers, self.auth, self.querystring )
        dataset = data.get_json ( )
        return dataset[ "problemOccurrence" ]

    def request_buildrun ( self , buildID ):
        self.request_url = self.host + self.api_url + "/builds/id:"  + str ( buildID )
        data = Metrics_Request ( self.request_url , self.headers, self.auth  )
        dataset = data.get_json ( )
        return dataset

    def process_build_results ( self , BuildId ):
        json_output = self.request_buildrun ( BuildId )
        if json_output != False:
            if "problemOccurrences" in json_output.keys ( ):
                for errorcode in self.request_problemOccurrences ( BuildId ):
                    Ci_CI_Build_Error = [
                        {
                            "measurement": "Ci_CI_Build_Errors" ,
                            "tags": {
                                "Product": "TeamCity" ,
                                "Server": "builds.ci.com" ,
                                "BuildID": 0 ,
                                "Project": "Test"
                            } ,
                            "time": "" ,
                            "fields": {}
                        }
                    ]
                    Ci_CI_Build_Error[ 0 ][ "tags" ][ "BuildID" ] = BuildId
                    Ci_CI_Build_Error[ 0 ][ "tags" ][ "Project" ] = json_output[ "buildType" ][ "projectId" ]
                    Ci_CI_Build_Error[ 0 ][ "time" ] = datetime.strptime ( json_output[ "finishDate" ] ,
                                                                               "%Y%m%dT%H%M%S+0000" ).strftime (
                        "%Y%m%dT%H%M%S+0000" )
                    Ci_CI_Build_Error[ 0 ][ "fields" ][ "error" ] = errorcode[ "type" ]
                    #dbclient = MetricsDB ( "localhost" , "8086" , "admin" , "admin" , "CiCIMetrics" )
                    #dbclient.insert_datapoint ( Ci_CI_Build_Error )
                    #del dbclient
                    print Ci_CI_Build_Error


            ##### Ci_CI_Build_Queue
            if "queuedDate" in json_output.keys ( ) and "startDate" in json_output.keys ( ) and "finishDate" in json_output.keys ( ):
                Ci_CI_Build_Queue = [
                    {
                        "measurement": "Ci_CI_Build_Queue" ,
                        "tags": {
                            "Product": "TeamCity" ,
                            "Server": "builds.ci.com"} ,
                        "time": "" ,
                        "fields": {}
                    }
                ]
                Ci_CI_Build_Queue[ 0 ][ "tags" ][ "Project" ] = json_output[ "buildType" ][ "projectId" ]
                Ci_CI_Build_Queue[ 0 ][ "time" ] = json_output[ "queuedDate" ]
                Ci_CI_Build_Queue[ 0 ][ "fields" ][ "BuildID" ] = BuildId
                Ci_CI_Build_Queue[ 0 ][ "fields" ][ "QueueToBuildStart" ] = (
                    datetime.strptime ( json_output[ "startDate" ] , "%Y%m%dT%H%M%S+0000" ) - datetime.strptime (
                        json_output[ "queuedDate" ] , "%Y%m%dT%H%M%S+0000" )).seconds
                Ci_CI_Build_Queue[ 0 ][ "fields" ][ "BuildStartToBuildEnd" ] = (
                    datetime.strptime ( json_output[ "finishDate" ] , "%Y%m%dT%H%M%S+0000" ) - datetime.strptime (
                        json_output[ "startDate" ] , "%Y%m%dT%H%M%S+0000" )).seconds
                Ci_CI_Build_Queue[ 0 ][ "fields" ][ "queuedDate" ] = datetime.strptime (
                    json_output[ "queuedDate" ] , "%Y%m%dT%H%M%S+0000" ).strftime ( "%Y%m%dT%H%M%S+0000" )
                Ci_CI_Build_Queue[ 0 ][ "fields" ][ "startDate" ] = datetime.strptime (
                    json_output[ "startDate" ] , "%Y%m%dT%H%M%S+0000" ).strftime ( "%Y%m%dT%H%M%S+0000" )
                Ci_CI_Build_Queue[ 0 ][ "fields" ][ "finishDate" ] = datetime.strptime (
                    json_output[ "finishDate" ] , "%Y%m%dT%H%M%S+0000" ).strftime ( "%Y%m%dT%H%M%S+0000" )
                Ci_CI_Build_Queue[ 0 ][ "fields" ][ "status" ] = json_output[ "status" ]

                #dbclient = MetricsDB ( "localhost" , "8086" , "admin" , "admin" , "CiCIMetrics" )
                #dbclient.insert_datapoint ( Ci_CI_Build_Queue )
                #del dbclient
                print Ci_CI_Build_Queue

            ##### Ci Build Result
            if "queuedDate" in json_output.keys ( ) and "startDate" in json_output.keys ( ) and "finishDate" in json_output.keys ( ):
                BuildResult = [
                    {"measurement": "Ci_CI_Build_Result" ,
                     "tags":
                         {"Product": "TeamCity" ,
                          "Server": "builds.ci.com" ,
                          "Project": "Demo" ,
                          "BuildID": "1234" ,
                          "status": "SUCCESSFUL"
                          } ,
                     "time": "2009-11-10T23:00:00Z" ,
                     "fields": {"status": "SUCCESSFUL"}
                     }
                ]

                BuildResult[ 0 ][ "time" ] = datetime.strptime ( json_output[ "queuedDate" ] ,
                                                                 "%Y%m%dT%H%M%S+0000" ).strftime (
                    "%Y%m%dT%H%M%S+0000" )
                BuildResult[ 0 ][ "tags" ][ "status" ] = json_output[ "status" ]
                BuildResult[ 0 ][ "tags" ][ "Project" ] = json_output[ "buildType" ][ "projectId" ]
                BuildResult[ 0 ][ "tags" ][ "BuildID" ] = BuildId
                BuildResult[ 0 ][ "fields" ][ "status" ] = json_output[ "status" ]
                #dbclient = MetricsDB ( "localhost" , "8086" , "admin" , "admin" , "CiCIMetrics" )
                #dbclient.insert_datapoint ( BuildResult )
                #del dbclient
                print BuildResult

##############################################################################
#	use below code to test various framwork classes
#
##############################################################################

#write_build_measurements()
#write_all_measurements()

def write_all_measurements():
    auth = Token("gitlab","ci_metrics")
    cred = auth.get_token()
    username, token = cred.popitem()
    host = "gitlab.bel.ci.lan"
    test = CiCIMetric_SCM_GitLab(host,token)
    result = test.get_projects_all ( )
    for project in result:
        test.write_commit_by_repository ( project , result[ project ] )
def write_build_measurements():
    auth = Token("teamcity","ci_metrics")
    cred = auth.get_token()
    username, token = cred.popitem()
    test = CiCIMetric_Build_TeamCity("builds.ci.com", username, token)
    for buildID in range(55000,55264 ):
        test.process_build_results(buildID)

    # get last build ID from Rediss
    # get pending builds from teamcity
    # get last sucessful build from teamcity
    #startBuildId = 55000
    #endBuildID = 55264
    #for BuildId in range ( startBuildId , endBuildID ):
    #    print "Processed Build Id: " , str ( BuildId )
    #    process_build_results ( BuildId )
def consul_test():
    c = Consul_Driver()
    c.set_last_build("LastBuild-Test/TeamCity-1","1")
    print c.get_last_build("LastBuild-Test/TeamCity-1")


consul_test()
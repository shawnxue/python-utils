#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import requests
import json
from requests.auth import HTTPBasicAuth
from xml.sax.saxutils import quoteattr


class TeamCity(object):
    def __init__(self, host=None, username=None, password=None, port=None, https=True):
        self.api_base_url = "httpAuth/app/rest"
        # add 'Accept': 'application/json' to headers to process a JSON response
        self.headers = {'Accept': 'application/json', 'Content-type': 'application/xml'}
        self.username = username
        self.password = password
        self.auth = HTTPBasicAuth(self.username, self.password)
        if https:
            self.host = 'https://' + host
        else:
            self.host = 'http://' + host
        if port:
            self.host = self.host + ':' + port

    def create_project(self, project_name=None):
        endpoint = "/projects/createProject"
        url = self.host + '/' + self.api_base_url + endpoint
        payload = {"data": {
            "name": project_name
            }
        }
        # json.dumps serialize python data structure (object) to string
        data = json.dumps(payload)
        result = self._post_teamcity(url=url, auth=self.auth, headers=self.headers, data=data)
        print result

    def delete_project(self):
        pass

    def create_vcs_root(self):
        pass

    def get_all_projects(self):
        endpoint = '/projects'
        url = self.host + '/' + self.api_base_url + endpoint
        result = self._get_teamcity_json(url=url, auth=self.auth)
        return result

    def get_all_users(self):
        endpoint = '/users'
        url = self.host + '/' + self.api_base_url + endpoint
        result = self._get_teamcity_json(url=url, auth=self.auth, headers=self.headers)
        return result

    def add_user_into_group(self, username=None, group_key=None):
        endpoint = '/users/username:' + username + '/groups'
        url = self.host + '/' + self.api_base_url + endpoint
        template = '<group key={key}/>'
        data = template.format(key=quoteattr(group_key))
        result = self._post_teamcity(url=url, auth=self.auth, headers=self.headers, data=data)
        return result


    def _get_project_by_name(self, project_name):
        project = None
        return project

    def _post_teamcity(self, url=None, auth=None, headers=None, data=None):
        requests.packages.urllib3.disable_warnings()
        response = requests.post(url=url, auth=auth, headers=headers, verify=False, data=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return False

    def _get_teamcity_json(self, url=None, auth=None, headers=None):
        requests.packages.urllib3.disable_warnings()
        response = requests.get(url=url, auth=auth, headers=headers, verify=False)
        if response.status_code == 200:
            return response.json()
        else:
            return None

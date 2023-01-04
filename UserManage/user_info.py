#!/usr/bin/env python3
# coding=utf-8
import time
import os
# 创建一个类来管理用户
class user_manage(object):

    def __init__(self, idlen=1e10):
        self.usrid = dict()
        self.idlen = int(idlen)
        self.pid = 0
        self.online = "online"
        self.offline = "offline"

    def add_new(self):
        timeid = int(time.time())
        num = 0
        while True:
            usrid = str(self.pid) +"_" +str(timeid) + "_" + str(num)
            if usrid in self.usrid:
                num += 1
            else:
                self.usrid[usrid] = dict()
                self.pid += 1
                self.pid = self.pid % (self.idlen)
                return usrid

    def del_usr(self, usrid):
        self.usrid.pop(usrid, "404")

    def in_usr(self, usrid):
        return usrid in self.usrid

    def man_usr(self, usrid, key=None, value=None):
        if key is not None and usrid in self.usrid:
            self.usrid[usrid][key] = value

    def add_usr_file(self, usrid, filepath):
        if usrid in self.usrid:
            if os.path.exists(filepath):
                if "filepath" not in self.usrid[usrid]:
                    self.man_usr(usrid, "filepath", [])
                self.usrid[usrid]["filepath"].append(filepath)
                return True
        return False

    def list_usr_file(self, usrid):
        if usrid in self.usrid:
            if "filepath" in self.usrid[usrid]:
                return self.usrid[usrid]["filepath"]
            else:
                return None
        else:
            return None
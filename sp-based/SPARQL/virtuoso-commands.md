## Virtuoso commands

### From README

The virtuoso backend will start up a web service, we can import our kb into it and then execute SPARQL queries by network requests. We install virtuoso in an Ubuntu 16.04 system. Following are specific steps.

1. Download and install virtuoso into our system.

```
git clone https://github.com/openlink/virtuoso-opensource.git Virtuoso-Opensource
cd Virtuoso-Opensource
git checkout stable/7
sudo apt-get install libtool gawk gperf autoconf automake libtool flex bison m4 make openssl libssl-dev
sudo ./autogen.sh
sudo ./configure
sudo make
sudo make install
```

2. Create a new user for virtuoso service

```
sudo useradd virtuoso --home /usr/local/virtuoso-opensource
sudo chown -R virtuoso /usr/local/virtuoso-opensource
```

3. Modify some necessary configs:

```
cd /usr/local/virtuoso-opensource/var/lib/virtuoso/db
sudo vim virtuoso.ini
```

Find the item `CheckpointInterval`, and change its value from default 60 to 0, to avoid automatical checkpoint process which will cause 404 error.

4. Start up the virtuoso service:

```
sudo -H -u virtuoso ../../../../bin/virtuoso-t -f &
```

Now you can access the service via the default port 8890.
Enter `[ip]:8890` in a browser, you will see the virtuoso service page.

5. Now we can import our kb into virtuoso. Before that, we need to convert our kb to `ttl` format and move it to proper position:

```
PYTHONPATH=$(pwd) python3 SPARQL/sparql_engine.py --kb_path datasets/kb.json --ttl_path datasets/kb.ttl
sudo chmod 777 datasets/kb.ttl
sudo mv datasets/kb.ttl /usr/local/virtuoso-opensource/share/virtuoso/vad
```

6. Enter the interactive terminal of virtuoso:

```
cd /usr/local/virtuoso-opensource/bin
sudo ./isql
```

7. Import our kb by executing these commands in terminal:

```
SPARQL CREATE GRAPH <http://nlp.project.tudelft.nl/kqapro>;
SPARQL CLEAR GRAPH <http://nlp.project.tudelft.nl/kqapro>;
delete from db.dba.load_list;
ld_dir('/usr/local/virtuoso-opensource/share/virtuoso/vad', 'kb.ttl', 'http://nlp.project.tudelft.nl/kqapro');
rdf_loader_run();
select * from DB.DBA.load_list;
exit;
```

`[graph_name]` could be any legal string, such as *KQAPro*.
You are success if `rdf_loader_run()` lasts for about 10 seconds.

### Additions

1. Display the process running virtuoso service: 

   ```shell
   ps -u virtuoso
   ```

   Then kill them by PID:

   ```shell
   kill -9 <PID>
   ```

2. Check the usage of port 8890:

   ```shell
   netstat -tulnp | grep 8890
   ```
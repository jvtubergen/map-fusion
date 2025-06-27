# Geoalg: Geometric algorithm master thesis


## Running experiments
Experiments are stored under `./src/measurements.py`.
To run experiments:
```python
experiment_one_base_table()
experiment_two_measure_threshold_values()
experiment_two_render_minimal_and_maximal_thresholds()
experiment_two_fusion_metadata()
experiment_three_sample_histogram()
```



## Data 
Data preparation steps (data modalities and unimodal inferrence graph results) can be bypassed by using git LFS to download raw (gps + sat) data and inferred (gps + sat) graphs.
The relevant portion for geoalg (the inferred sat and gps graphs alongside the ground truth osm) are stored under `./data/graphs/(gps|osm|sat)`.

Note the region of interest is decided by the GPS data and is limited to the cities Berlin and Chicago for practical reasons.

Related dependencies are:
* gps: [roadster]()
* sat: [gmaps-image]() and [sat2graph-simplified]()
* osm: [OSMnx]()
Both for sat and osm packages are already included by preparing virtual environment with [hatch]().
For GPS relies on Java and has to be dealt with manually.

In case you want to run through this unimodal data inferrence pipeline:
* The OSM graph data is retrieved by performing API calls to [OpenStreetMaps]() API by using the [OSMnx]() library.
    1. Function `./src/data/osm:download_data` to download graph data and store it to `./data/graphs/sat/(berlin|chicago)/(vertices.txt|edges.txt)`.
* The SAT image data is retrieved by first obtaining images with [gmaps-image]() and then inferring them with [sat2graph-simplified]().
    1. Function `./src/data/sat.py:download_data` calls [gmaps-image]() to obtain satellite image of Berlin and Chicago (500MB and 77MB respectively) and stores the images at `./data/sat/images/(berlin|chicago)` with the upper-left pixel-coordinate and the related zoom level (with scale applied) add as image metadata. Note it requires a Google API key written to `~/.cache/gmaps-image/api_key.txt` to perform API calls.
    2. Function `./src/data/sat.py:infer` calls [sat2graph-simplified]() to convert the resulting images of step 1 into inferred graphs, storing these at `./data/graphs/sat/(berlin|chicago)/(vertices.txt|edges.txt)`.
* The GPS data is limited to the cities of Berlin and Chicago in [mapconstruction]() dataset. 
    1. Function `./src/data/gps.py:download_data` to download and extract [mapconstruction]() dataset into `./data/gps/traces/(berlin|chicago)`. 
    2. To infer, clone [roadster]() repo, run the Java code on the two cities, and store the inferred graph data to `./data/gps/inferred/(berlin|chicago)/(vertices.txt|edges.txt)`.  For gps (roadster) clone the repo in `./external/roadster` with `gps clone  ./external/roadster`, get the code running, and perform the command `bash ./src/data/gps/infer.sh` to use [javac]() for inferring both on Berlin and Chicago.
    3. Convert the inferred graph data from UTM to WSG coordinates by running the function `./src/data/gps.py:convert` that reads in the graph data of step 2 and writes it to `./data/graphs/gps/berlin|chicago)/(vertices.txt|edges.txt)`.


## Preparation


### Uni-modal data inferrence
Use [roadster]() (see #roadster) to 

## Folder structure

* data: content related to inferrence, experiments, visualizations
    * gps: data for inferrence pipeline unimodal GPS-based map reconstruction method
        * traces: raw unimodal data. One folder per city with a text file for every track of traces.
        * inferred: output of Roadster. 
    * sat: data for inferrence pipeline unimodal SAT-based map reconstruction method
    * graphs: inferred graphs

* src: code
    * data: pre-processing to obtain unimodal input data and the unimodal inferrence pipeline 
        * sat
            * 
        * gps
        * osm
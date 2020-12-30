---
title: Tutorial on high computation Vaex package and interactive dashboards with Vaex and Dash
date: 2020-11-18
excerpt_separator: <!--more-->
---
Dashboards are integrated elements with the analytics and insights extracted from the dataset, which extract the insightful findings from data features, produce business metrics, or track the performance of a model in production. The dashboard would provide the insights from the data extraction, and show the value for the organization. <br>
<!--more-->
**In this article, you will learn the singular value decomposition and truncated SVD of the recommender system:** <br>
(1) Introduction <br>
(2) Walk through on Vaex <br>
(3) Methods to handle large dataset in Vaex <br>
(4) Dash & Vaex <br>
(5) Hands-on experience with python code <br>
(6) Dashboard Tutorial in Dash and Vaex package <br>
(7) Diverse Visualization plots <br>
## Walk through on Vaex
Vaex is a python library to process the large tabular datasets for visualization and produces the **Out-of-Core Dataframes** (similar to Pandas). The computation supports the statistics calculations such as mean, sum, count, and standard deviation. With its computation resources, Vaex computes the billion objects/rows per second on the N-dimensional grid. In terms of visualization, interactive visualization of big data such as **histograms**, **density plots**, and **3d volume** rendering is also supported by Vaex. To optimize the performance, Vaex uses memory mapping, a zero memory copy policy, and lazy computations. Vaex exports the file in HDF5 format, flexible for other programming languages. Vaex is a visualization tool to generate graphs and explores large tabular datasets. With the 1 and 2d data input, Vaex extracts richer information from the subspaces of the columns (dimensions) analysis. <br>
DataFrame is the class (data structure) in vaex, and is generated from the input of different data files. In vaex, the function open() is to open a file. With the open function, Vaex is able to connect to a remote server. <br>
Alternatively, Vaex can read the data remotely from Amazon’s S3, and renders an HDF5 file. Hence, the data is lazily downloaded and cached to the local machine. Regarding “Lazily downloaded”, only portions of the data is downloaded. For instance, there is a large dataset with 100 columns and 5000 rows. The first and last 5 rows of data would be downloaded via `print(df)`. With the plots generated from only 5 columns, these columns will be downloaded and cached to the local machine. By default, data is cached at the directory `$HOME/.vaex/file-cache/s3` from the stream of S3. The access to the cache directory is as fast as the native disk. `profile_name` argument is to use a specific S3 profile, and the file is saved in `s3fs.core.S3FileSystem`. <br>
In the S3 URL HTML, the parameters are introduced in S3 options: <br>
- anon: anonymous access or not (false by default). (Allowed values are: true,True,1,false,False,0)
- se_cache: Use the disk cache or not, only set to false if the data should be accessed once. (Allowed values are: true,True,1,false,False,0)
- profile_name and other arguments are passed to s3fs.core.S3FileSystem <br>
These arguments are passed as kwargs, but the argument such as anon passed as a boolean, not a string.


## Methods to handle large dataset in Vaex
The common challenge that many organizations face is a large amount of data with the range of millions of rows of data, which is overwhelming to be processed. Data scientists face difficulty to work with large datasets, while most tools are not fit to process the data with such scale. It’s more challenging to build an interactive dashboard with the input of a large-scale dataset. Vaex, an Open Source DataFrame library in Python, enables us to work with a large dataset. Vaex supports memory mapping, and it would not be cached in RAM all at once. Through memory mapping, the same physical memory is shared amongst all processes. Such function is quite useful in Dash, which supports workers to scale vertically and Kubernetes to scale horizontally. Besides, Vaex processes the large dataset with efficient, fully parallelized out-of-core algorithms. The API shares a similar foundation set by Pandas.

## Dash & Vaex
Vaex works along with Dash to build simple, and interactive analytical dashboards or web applications. Dash applications support reactive functions. With the users’ interactions of pushing a button or moving a slider, the callbacks are implemented on the server, which updates the application via the computation. With the stateless server, there is no memory required from the users’ interaction. Dash can both scale vertically with more workers and nodes. With the stateless function, Vaex computes the dataset such as filtering along with aggregating computation, and it processes the request instead of modifying or copying the data. Vaex produces a small result from the computation or group-by aggregations for visualizations since they will be transferred to the browser. <br>
In addition, Vaex can process each request on a single node or worker within a short period, and it’s not required to set up a cluster. Distributed computing is another tool to tackle larger datasets. The article will introduce how to build an interactive web application with the input of a large dataset that barely fits into RAM on most machines (12 GB). Data manipulation, aggregation, and statistic computations are done through Vaex. Then, the plots would be visualized interactively through Plotly and Dash.
## Hands-on experience with python code
The dataset used for the dashboard is New York Taxi, which is a public dataset to showcase the way for data manipulation with its relatability and size. The data contains 100 million trips over a full year of the YellowCab Taxi company. The interactive web dashboard shows the estimated cost and duration of their next trip to the prospective passengers with Dash and Vaex, while the web application would show the general trend of the taxi routes. <br>
The public availability, relatability, and size have made the New York Taxi dataset the de facto standard for benchmarking and showcasing various approaches to manipulating large datasets. The following example uses a full year of the YellowCab Taxi company data from their prime, numbering over 100 million trips. We used Plotly, Dash, and Vaex in combination with the taxi data to build an interactive web application that informs prospective passengers of the likely cost and duration of their next trip, while at the same time giving insights to the taxi company managers of some general trends.

## Dashboard Tutorial in Dash and Vaex package
We’ll walk through the tutorial to go over the dashboards with some functions provided with the input of the data that barely fits in memory with Dash and Vaex. The application, trip planner, enables users to select the pick-up locations in New York City in the interactive heatmap. The interactive map supports the pan and zoom function, and the map would be updated via recomputation after each action. The user can click on the map to select the origin and destination. Then, the dashboard would pop out the cost and duration based on the designated routes. Furthermore, the user can specify the day and hour range to gain detailed information about the trip. <br>
Vaex would memory-map the data and input part of the data no matter how large the data is. While multiple workers are running in the dash application, each of them would be distributed with an equal memory-mapped file. <br>
Next, the layout of the Dash application would be introduced <br>
The next step is to set up the Dash application with a simple layout. In our case, these are the main components to consider: <br>
- The components part of the “control panel” that lets the user select trips based on time dcc.Dropdown(id='days') and day of week dcc.Dropdown(id='days');
- The interactive map dcc.Graph(id='heatmap_figure');
- The resulting visualizations are based on the user input, which will show the distributions of the trip costs and durations, and a markdown block showing some key statistics. The components are dcc.Graph(id='trip_summary_amount_figure'), dcc.Graph(id='trip_summary_duration_figure'), and dcc.Markdown(id='trip_summary_md') respectively.
- Several dcc.Store() components track the users’ state at the client-side. <br> <br>
<script src="https://gist.github.com/denisechendd/76b11040fee5fbca6ab04564453742bc.js"></script>
**Pic1: Code cell of the dash function of pick-up days and hour slider** <br>
Now let’s talk about how to make everything work. We organize our functions into three groups: <br>
- compute_ functions are the basis for the visualization to calculate the relevant aggregations and statistics
- create_figure_ function creates the figure from the aggregation compute
- Dash callback functions get to interact with the compute function, and transfer the output to the figure creation functions after the user makes changes. <br>
The separated functions into three groups would better organize the dashboard functionality. Besides, the application can be pre-populated and avoids the callback triggering on the initial page load. <br>
From the heatmap, the user can select the pick-up hour and day of the week from the Range Slider and Dropdown elements as the data subset. <br>
Let’s start by computing the heatmap. The initial step is selecting the relevant subset of the data the user may have specified via the Range Slider and Dropdown elements that control the pick-up hour and day of the week respectively: <br> <br>
<script src="https://gist.github.com/denisechendd/d51aeef6e2bd94c7eda6a2967825d6ef.js"></script>
**Pic2: Code cell of heatmap function** <br>
From the code above, the data frame is copied into the selection function, which is the stateful object in the DataFrame. Dash is multi-threaded, and it uses the selection method other than filtering to boost the performance. The code cell below computes the heatmap data: <br> <br>
<script src="https://gist.github.com/denisechendd/3dc0f4bfdc61637334dce040364a4060.js"></script>
**Pic3: Code cell of the heatmap compute function** <br>
All Vaex DataFrame methods are applied to all sizes of data with its parallelization and out-of-core functions. From the heatmap computation, two columns are passed via the `binby` argument to the `.count()` method. Then, the number of samples is calculated in a grid specified by those axes. The grid is drawn from two elements `shape` (i.e. the number of bins per axis) and `limits` (or extent). The output of the data array is `array_type="xarray"`, where the numpy array has the labeled dimension. The numpy array is convenient for plotting. <br>
With the heatmap computation, the code cell below will show how to create the figure on the dashboard. <br> <br>
<script src="https://gist.github.com/denisechendd/bc1209ab148382e3c27265b04750260f.js"></script>
**Pic4: Code cell of the heatmap plot**  <br>
From the function above, the function of Plotly Express is applied to render the heatmap. Given the `trip_start` and `trip_end` coordinates, both variables would be added as individual `plotly.graph_objs.Scatter` traces to the figure. The interactive Plotly figure supports the zooming, panning, and clicking functions. <br>
The code cell below shows how to update the heatmap figure from the modifications made in the data selection or changes to the map view using the Dash callback. <br> <br>
<script src="https://gist.github.com/denisechendd/f89b720e260e522082c686ac3357572e.js"></script>
**Pic5: Code cell of the heatmap update** <br>
From the code block above, the function would be called when there is a change of the `Input` values. In the function, `compute_heatmap_data` is to do the aggregation computation with the new input parameters, and the new heatmap figure is generated with the computed result. `prevent_initial_call` argument of the decorator is to avoid the function from being called when the dashboard runs in the first round.
Despite the fact that `trip_start` or `trip_end` parameters don’t appear in `compute_heatmap_data`, `compute_heatmap_data` is called when both parameters change the input, and `update_heatmap_figure` is triggered. The decorator attached to `compute_heatmap_data` is to prevent several calls of the function. `flask_caching` library, suggested in Plotly, is fast, easy, and simple to cache old computations for 60 seconds. <br>
The code cell below shows the user interactions with the heatmap via panning and zooming from the Dash callback function. <br> <br>
<script src="https://gist.github.com/denisechendd/12b42e3fd6db69bbdb8b918e8b150a54.js"></script>
**Pic6: Code cell of heatmap pan and zoom function** <br>
According to the dash callback below, it is to capture and respond to click events: <br> <br>
<script src="https://gist.github.com/denisechendd/569ac9d40bf187832837ca3d54998ebf.js"></script>
**Pic7: Code cell of capturing the click events of heatmap** <br>
Update key components in both the above callback functions are to render the heatmap. Therefore, when there are events like click or relay (pan or zoom), update_heatmap_figure function would be called from the updating key components, and it would update the heatmap figure. The function above creates the fully interactive heatmap figure. The heatmap would be updated via external controls such as the RangeSlider and Dropdown menu or through the interactive function in the figure. <br>
Since the Dash application is stateless, reactive, and functional, functions in Dash are to create visualizations. In the Dash application, we can click and select trips starting from the “origin” and ending at the “destination” point. Regarding those trips, it would show up the cost distribution and duration, and highlight the most likely values for both. These can be coded in the function below. <br> <br>
<script src="https://gist.github.com/denisechendd/0e5c00e025eed43f21859dff52b8b52c.js"></script>
**Pic8: Code cell of the heatmap details of trip duration and cost** <br>
The helper function is defined to create the histogram figure with the input of the aggregated data. <br> <br>
<script src="https://gist.github.com/denisechendd/3033d7e3d092c576f4a63303842749b3.js"></script>
**Pic9: Code cell of the histogram regarding the cost amount and trip duration** <br>
With all the components, we can link them to the Dash application via a callback function: <br> <br>
<script src="https://gist.github.com/denisechendd/d525770fc1112705d446a8fd24a8962d.js"></script>
**Pic10: Code cell of the trip summary regarding number of rides, total cost, and trip duration** <br>
The callback function above is updatable to the changes from the control panel, along with the click selection of new origin or destination points. Through the registered event, the callback function is triggered and will call the compute_trip_details and create_histogram_figure functions with new parameters input. Then, the visualization is updated with the values input to these functions. <br>
There is one condition considered when a user only selects the starting point, but not yet click on the new destination. Therefore, the histogram would be “blank out” with the functions below. <br> <br>
<script src="https://gist.github.com/denisechendd/13fbd350b2b1602e6822b65dcde6d6a5.js"></script>
**Pic11: code cell of the trip origin and destination function in heatmap** <br>
Finally, the code in the source file below is to run the dashboard. <br> <br>
<script src="https://gist.github.com/denisechendd/8b8cd3c035773f6d6bd5939e0295bd12.js"></script>
**Pic12: Code cell of the Vaex dashboard source file** <br>
Then, the interactive Dash application is created! After downloading the taxi data from the github page, the dashboard would be executed locally through the source file via the command line of `python app.py` in the terminal. <br>
And there we have it: a simple yet powerful interactive Dash application! To run it locally, you can execute the `python app.py` command in your terminal, provided that you have named your source file as “app.py”, and you have the taxi data at hand. You can also review the entire source file via this GitHub Gist.

## Diverse Visualization plots
There are many visualization plots in Plotly. Apart from the typical heatmaps and histograms, the dashboard includes several interactive, but not quite common methods to show the aggregated data visualization. On the first tab, there is a geographical map colored by the number of taxi pick-ups in NYC zones. A user is able to select the pickup and destination place on the map and gain information on popular destinations (zones and boroughs) via the Sankey and Sunburst diagrams. This functionality is created in the same way as the above Trip planner tab. The core of these functions is applied with the groupby operations to format the data to meet the Plotly requirements. The code is referenced from the Github.


## In Conclusion
- Vaex is a python library to process the large tabular datasets for visualization and produces the **Out-of-Core Dataframes** (similar to Pandas). In terms of visualization, interactive visualization of big data such as **histograms**, **density plots**, and **3d volume** rendering is also supported by Vaex. To optimize the performance, Vaex uses memory mapping, a zero memory copy policy, and lazy computations. Vaex exports the file in HDF5 format, flexible for other programming languages. Vaex is a visualization tool to generate graphs and explores large tabular datasets.
- Vaex supports memory mapping, and it would not be cached in RAM all at once. Through memory mapping, the same physical memory is shared amongst all processes. Such function is quite useful in Dash, which supports workers to scale vertically and Kubernetes to scale horizontally. Besides, Vaex processes the large dataset with efficient, fully parallelized out-of-core algorithms.
- Vaex works along with Dash to build simple, and interactive analytical dashboards or web applications. Dash applications support reactive functions. With the users’ interactions of pushing a button or moving a slider, the callbacks are implemented on the server, which updates the application via the computation. With the stateless server, there is no memory required from the users’ interaction. Dash can both scale vertically with more workers and nodes. With the stateless function, Vaex computes the dataset such as filtering along with aggregating computation, and it processes the request instead of modifying or copying the data.

## Reference
- Github - dash-120million-taxi-app <br>
https://github.com/vaexio/dash-120million-taxi-app
- Interactive and scalable dashboards with Vaex and Dash <br>
https://medium.com/plotly/interactive-and-scalable-dashboards-with-vaex-and-dash-9b104b2dc9f0

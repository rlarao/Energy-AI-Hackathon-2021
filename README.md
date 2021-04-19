# Energy A.I. Hackathon 2021

## Hosts: [Prof. Michael Pyrcz](https://twitter.com/GeostatsGuy) and [Prof. John Foster](https://twitter.com/johntfoster)

### Architects: [Honggen Jo](https://twitter.com/HonggeunJ) and Mingyuan Yang

### Sponsor: [Prof. Jon Olson](https://twitter.com/ProfJEOlson), and the [Hildebrand Department of Petroleum and Geosystems Engineering](https://twitter.com/UT_PGE)

### Organization and Student Engagement: Gabby Banales and Sara Hernando

___

### Energy A.I. Problem Description 

**Goal**: develop a data analytics and machine learning workflow in Python to predict estimates and uncertainty models for cumulative oil 3 year production for **10 unproduced wells**. 

#### Background

We challenge the Energy A.I. hackathon teams of The University of Texas at Austin, engineering and science Students to build a data-driven model to predict oil production. This will require:

* data analysis and evaluation of multiple data sources and a variety of features
* integration of domain expertise
* training and tuning robust machine learning prediction models
* integration and modeling of uncertainty  

This data-driven approach will replace the conventional engineering and geoscience approach:

* characterizing and modeling the subsurface
* physics-based fluid flow simulations

___
 
#### The Reservoir Unit

Specifications of the reservoir unit of interest: 

* **Depositional Setting**: clastic deepwater reservoir unit with extents 10km by 10km by 50m (thickness)
* **Fluids**: initial oil water contact is at the depth of 3067.4m and the average connate water saturation is about 20.3%
* **Structure**: Anticline structure with a major vertical fault that crosses the reservoir (see location and equation on the image below). 
* **Grids**: the 2D maps conform to the standard Python convention, origin on Top Left (see the image below).

<img src="https://github.com/PGEHackathon/data/blob/main/image.png" width="600" height="400">

* **Wells**: 83 vertical wells were drilled across the reservoir and completed throughout the payzones. Due to the field managements, only 73 wells were open to produce oil for the first three years, while the remaining 10 wells were kept shut-in. At the end of the third year, the remaining 10 unproduced wells are planned to be openned.

* **Question**: What will be the the cumulative oil production for each of these 10 unproduced (preproduction) wells over the next 3 years?  

___

### Available Data Files Inventory

You have the following data files available.

#### Well Logs

These two files contain the well log data along the wellbore for all 83 wells.

* **wellbore_data_producer_wells.csv** - well logs for the previous production wells, well indices from 1 to 73
* **wellbore_data_preproduction_wells.csv** - well logs for the remaining, preproduction wells, well indices from 74 to 83 

Comments: 

* all the well names are masked (replaced with simple indices from 1 to 83) and coordinates transformed to the area of interest to conceal the actual reservoir. 
* available petrophysical and geo-mechanical properties are listed. 
* blank entries in the file indicate missing data at those locations.

#### Map Data

The following map data are available:

* **2d_ai.npy** - acoustic impedance (AI) inverted from geophysical amplitudes and interpretations
* **2d_top_depth.npy** - depth of the reservoir unit top mapped from interpreted geophysical data
* **2d_sand_propotion.npy** - proportion of sand facies over the vertical column, 2D facies proportion map
* **2d_sandy_shale.npy** - proportion of sandy shale facies over the vertical column, 2D facies proportion map
* **2d_shaly_sand.npy** - proportion of shaly sand facies over the vertical column, 2D facies proportion map
* **2d_shale.npy** - proportion of shale facies over the vertical column, 2D facies proportion map

Comments:

* 2D maps are regular grids 200 by 200 cells, cell extents are 50m by 50m, extending over the reservoir 
* values indicate the vertically averaged property, vertical resolution is the entire reservoir unit
* the indices follow standard Python convention, original is top left corner, indices are from 0, 1, ..., n-1 and the first index is the row (from the top) and the second index is the column from the left.
* e.g. to select the 5th grid cell in x (column) and the 10th grid cell in y (rows), use ndarray[9,4] in Python (aside, array[10,5] in Matlab). 
* the origin of the 2D data (e.g., array[0,0]) is the center of the top left cell, 25m and 25m along y and x direction (refer to the image above)

#### Production History

The following production history is available:

* **production_history.csv** - the cumulative oil and water productions for the 73 previous production wells after 1 year, 2 years and 3 years.
___

### Required Hackathon Submissions

By April 18th at noon each team must submit:

* **Solution Table** - a .csv file with your predictions for the 10 preproduction wells, estimates and uncertainty model realizations. The submitted file should follow the format of the provided template [solution.csv](https://github.com/PGEHackathon/data/blob/main/solution.csv) for automatic scoring.

    * the file must be named 'solution.csv' with final values in a commit and then pushed to Github for the automated scoring.

* **Python Workflow and Associated Files** - commited to this repository with the workflow as a Jupyter Notebook .ipynb file along with all data files required to reproduce your team's solutions. The submitted workflow Jupyter Notebook should follow the format of the provided template [Hackathon_ProjectTemplate](https://github.com/PGEHackathon/resources/blob/main/Hackathon_ProjectTemplate.ipynb) for enhanced workflow communication and code readibility.

* **Presentation** - a PowerPoint slide deck .PPTX file for your team's final presentation to our judges. The submitted presentation should follow the format of the provided example presentation [Hackathon_PresentationTemplate](https://github.com/PGEHackathon/resources/blob/main/Hackathon_PresentationTemplate.pptx).

The Workflow and Presentation submission templates are in the [resources respository](https://github.com/PGEHackathon/resources) and the results submission template is in this repository.

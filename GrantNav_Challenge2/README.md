# 360Giving Data Visualisation Challenge - Part 2
This work was part of the Data Visualisation Challenge hosted by 360Giving https://challenge.threesixtygiving.org/. It looks to answer the following question posed by a panel of experts in the grantmaking sector: <i>User-led organisations: Who funds them, in what thematic area, how much funding do they receive and what type of organisation are they (e.g. CIC, charity, co-operative, community group)? </i>

A snapshot of the visualisation is shown below. The link to the Interactive visualisation can be found found <a href = "https://ryankarlos.github.io/GrantNav_Challenge2/">here</a>

<p>
    <img src="https://github.com/ryankarlos/GrantNav_Challenge2/blob/gh-pages/screenshot.jpg" width="1000" height="800" />
</p>


## DATA COLLECTION AND CLEANING 

The data was acquired from the Grant Nav database filtered from years 2008-2017, with user-led organisations filtered out using the 'user-led' search term. The data was subsequently cleaned using a custom script in R programming language to remove unnecessary columns (like additional redundant location info) and adapting formats of date The data was then filtered into themes according to keywords (note that initially a range of themes were tested to see how much data was included in each – four theme were chosen 
* ‘elderly’, ‘older people’, ‘dementia’ would suggest the theme is elderly 
* projects with ‘sports’, ‘fitness’ for sports theme 
* projects with ‘science’ for science theme 
* projects with 'environment' for environment theme 

The major challenge was to manually fill in location data for recipient organisations which did not have any location info on Grant Nav (a large proportion of these were universities). For this, a combination of websites were used to check the organisation locations like Find that Charity, Charity Base and the Charity Commission. 

## VISUALISATION 

The visualisation was built in D3js. I decided to display the data using a Sankey diagram and force directed network graph, both of which are good for viewing flow of information from funders to user led organisations as well as visualising how densely connected organisations are. For the Sankey diagram, I have only included funders who have donated at least £10000 and above to allow the visualisation to be deciphered easily, when viewing all the organisation types. The force directed graph however does not have any such restriction and includes information about all funders and user led organisations for the specific year and theme. The liquid gauges indicate the number (which can also be switched to amount in £) of each type of organisation in the whole dataset. Clicking on any of the gauges filters the force directed graph and sankey diagram according to the respective organisation type . Clicking again removes the filter and returns it to the default graph with all organisation types (if an organisation type is selected , the rest of the gauges will be blurred out). 

Hovering over the Sankey diagram links shows the funder, receipient and amount awarded in the tooltip. Placing the mouse cursor on any funder node will produce a colour flow from that node to each of the recipients. Similarily, hovering over any of the nodes of the forced directed graph and will show the organisation (im working on reoresenting the amount awarded).

## LICENSING

As required as part of this challenge, this work is licensed under a Creative Commons Attribution 4.0 International License (included in the footer of the dashboard). 

## RESOURCES

* Find that charity http://apps.charitycommission.gov.uk/showcharity/registerofcharities/RegisterHomePage.aspx 
* Charity Base https://charitybase.uk/ 
* Charity Commission Register of Charities http://apps.charitycommission.gov.uk/showcharity/registerofcharities/RegisterHomePage.aspx

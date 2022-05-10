$(document).ready(function()

{
	var  map_width    =  $('#map').width();
	var  map_height   =  $('#map').height();
	var  bar_width    =  $('#bar').width();
	var  bar_height   =  $('#bar').height();
	 var legend_width    =  $('#legend').width();
	 var legend_height   =  $('#legend').height();
   var centered;
	 var  scale = 1.0;
	var parseDate;
	var formatDate;
	var coordinates = [];
	var tweets = [];
	var month = [];
	var data_month = [];

	console.log(map_width)
	console.log(map_height)

$("#lda").css('display','none');
$("#LDA-description").css('display','none');
d3.select("#bar-freqs").selectAll("path").style("fill", "none").style("stroke", "none");

$(".button2").click(function () {
				  $(".leftcolumn").css('display','none');
					$(".rightcolumn").css('display','none');

					$("#lda").show()
					$("#LDA-description").show();
});


$(".button1").click(function () {
					$(".leftcolumn").show();
					$(".rightcolumn").show();
					$("#lda").css('display','none');
					$("#LDA-description").css('display','none');
});


// Set tooltips
var tip = d3.tip()
            .attr('class', 'd3-tip')
            .offset([-1, 100])
            .html(function(d) {
              return "<strong>Country: </strong><span class='details'>" + d.properties.name;
            })

  var projection = d3.geo.mercator().scale(140).center([0,5]).translate([map_width / 2, map_height / 1.8]);  /// for drawing the map

	var path = d3.geo.path().projection(projection);


	var zoom = d3.behavior.zoom()
    .scale(scale)
		.scaleExtent([1, 5])
		.on("zoom", zoomed);

		/// Add map svg ////
		function zoomed() {
		  g.attr("transform", d3.event.transform);
		}

var svg_map = d3.select('#map').append("svg")
                .attr("width", map_width)
                .attr("height", map_height)
                .call(zoom);

var g_map =  svg_map.append('g').attr('class', 'map');

	g_map.call(tip);

	var color = d3.scale.linear()
		.domain([])

	var dataArray;

	parseDate  = d3.time.format("%d/%m/%Y").parse;
	formatDate = d3.time.format("%b")

	function zoomed() {
	  var translateX = d3.event.translate[0];
	  var translateY = d3.event.translate[1];
	  var xScale = d3.event.scale;
	  g_map.attr("transform", "translate(" + translateX + "," + translateY + ")scale(" + xScale + ")")
	}

  function reset() {
    scale = 1.0;
    g_map.attr("transform", "translate(0,0)scale(1,1)");
    zoom.scale(scale)
        .translate([0,0]);
  }

	//var map_title =  d3.selectAll('#map-title svg')
	//				  .attr('viewBox', '0 0 ' +  ( maptitle_width) + ' ' + ( maptitle_height ) )
	//				  .attr('height', '100%')
	//				  .attr('width', '100%')
	//				  .append('g')


	d3.select("#zoom-button").on("click", reset);

	 // load and display the World
	 // world topojson file got from http://bl.ocks.org/mbostock/raw/4090846/world-50m.json

	d3.json("world-topo-min.json", function(error,topology){
				if(error) throw error;

				g_map.selectAll(".country")
				  .data(topojson.feature(topology, topology.objects.countries).features)
				  .enter()
				  .append("path")
				  .attr("d", path)
				  .on('click', clicked)
				  .attr("id", function(d,i) { return d.id; })
				  .attr("title", function(d,i) { return d.properties.name; })
				  .style("fill", "#ccc")
				  .style("stroke", "grey")
				  .style("stroke-width", 1)
				  .on("mouseover", function(d){
					      tip.show(d);
						    d3.select(this)
							    .style("fill", "yellow");
					 })
					.on("mouseout", function(d){
					    	tip.hide(d);
					    	d3.select(this)
						      .style("fill", "#ccc");
					 });

			// function for clicking and zooming into country
				function clicked(d) {
						  var x, y, k;
						  if (d && centered !== d) {
							var centroid = path.centroid(d);
							x = centroid[0];
							y = centroid[1];
							k = 3;
							centered = d;
						  } else {
							x = map_width / 2;
							y = map_height / 2;
							k = 1;
							centered = null;
				  }

				  g_map.selectAll("path")
					  .classed("active", centered && function(d) { return d === centered; });

				  g_map.transition()
					  .duration(750)
					  //.attr("transform", "translate(" + map_width/2 + "," + map_height/2 + ")scale(" + k + ")translate(" + -x + "," + -y + ")")
					  .style("stroke-width", 1.5 / k + "px");
				}


			  var fav_colors = ["red","blue", "green"]
			  //var fav_colors = colorbrewer.PuRd[3].reverse()

			  var legend =  d3.select('#legend')
				                .append("svg")
								        .attr('height', '100%')
								        .attr('width', '100%')


			var svg_legend = legend.append('g')

				var grad = svg_legend.append('defs')
									.append('linearGradient')
									.attr('id', 'grad')
									.attr('x1', '0%')
									.attr('x2', '0%')
									.attr('y1', '89%')
									.attr('y2', '0%');

				grad.selectAll('stop')
					.data(fav_colors)
					.enter()
					.append('stop')
					.attr('offset', function(d, i) {
								 return ((i/fav_colors.length))*100 + '%';
					})
					.style('stop-color', function(d) {
									  return d;
					})
					.style('stop-opacity', 0.9);

				svg_legend.append('rect')
				  .attr('x', 0)
				  .attr('y', 0)
				  .attr('width', "100%")
				  .attr('height', "100%")
				  .attr('fill', 'url(#grad)');

				  var svg_bar =  d3.select('#bar').append("svg")
								  .attr('height', '100%')
								  .attr('width', '100%')
								  .append('g')
								  .call(zoom)

					svg_bar.attr("transform", "translate(150,0)");

				var bar_g = svg_bar.append("g")

					//The code to load the cities should be placed inside the function that is loading the World map.
					// This is to avoid the city/country data being before the world data if the world data takes too
					// long to load. This way the city data doesn't get loaded until the World data is loaded and then
					// the circles get drawn on top of the world instead of under it.


			var div = d3.select("body").append("div")
														    .attr("class", "tooltip")
														    .style("opacity", 0);

				function barupdate(data, select){

								 bar_g.selectAll("rect").remove();
								 bar_g.selectAll("text").remove();



					//select all bars on the graph, take them out, and exit the previous data set.
					//then you can add/enter the new data set
								var bar  =  bar_g.selectAll("rect")
																	 .data(data)
																	 .enter()
																	 .append("rect")
																	 .attr("class", "bar")
																	 .attr("height", 0)

									 bar.transition()
											 .duration(1000)
											 .attr("height", function(d, i) {
																if (select == "Favourites") {
																		 return (Math.sqrt(+d.Favourite_count)*10 )
																 }
																 else {
																		return (Math.sqrt(+d.Retweet_count)*10 )
																 }
													 })
											 .attr("width","6%")
											 .attr("x", function(d, i) {return (i * (bar_width/9)) })
											 .attr("y", function(d, i) {return bar_height/8 })
											 .style("fill", function(d){
																 if (select == "Favourites") {
																			return "green"  // green
																	}
																	else {
																		 return "red"
																	}
											 });

							 // Select, append to SVG, and add attributes to text
								 bar_g.selectAll("users")
											 .data(data)
											 .enter().append("text")
											 .transition()
											 .duration(1000)
											 .text(function(d) {return (d.User)})
												.style("font-size", "13px")
												.attr("class", "text")
												.attr("x", function(d, i) {return (i *  (bar_width/9 -1))})
												.attr("y", function(d, i) {
																		if (select == "Favourites") {
																				 return (Math.sqrt(+d.Favourite_count)*10 + 85)
																		 }
																		 else {
																				return (Math.sqrt(+d.Retweet_count)*10+ 85)
																		 }
												 });

									bar_g.selectAll("label")
												.data(data)
												.enter().append("text")
												.transition()
												.duration(1000)
												.text(function(d) {
																if (select == "Favourites") {
																		return (d.Favourite_count)
																 }
																 else {
																		return (d.Retweet_count)
																 }
												 })
												.style("fill", "white")
												.style("font-size", "20px")
												.attr("class", "text")
												.attr("x", function(d, i) {return (i * (bar_width/9) + 10)})
												.attr("y", function(d, i) {
																	if (select == "Favourites") {
																		 return (Math.sqrt(+d.Favourite_count)*10 + 60 )
																	 }
																	 else  {
																			 return (Math.sqrt(+d.Retweet_count)*10 + 60 )
																		}
														});

								/// Adding in the title of the graph
									bar_g.append("text")
												 .attr("x", "49%")
												 .attr("y",bar_height/1.5)
												 .attr("text-anchor", "middle")
												 .transition()
												 .duration(1000)
												 .style("font-size", "20px")
												 .style("font-weight", "bold")
												 .text("Tweets with the most " + select);


								 bar.on("mouseover", function(d) {		/// tooltip on mouseover
														 div.transition()
															 .duration(200)
															 .style("opacity", 0.9);
														 div.html("<br/>" + "Tweet: "  + d.Text + "<br/>" + "<br/>" + "Location: "  + d.Location + "<br/>" + "Date: "  + d.Date)
															 .style("left", (d3.event.pageX - 460) + "px")
															 .style("top", (d3.event.pageY - 28) + "px")
															 .style("width", "25%")
															 .style("height", "25%")
															 .style("font-size", "14px");
														 })
										 .on("mouseout", function(d) {
															 div.transition()
																.duration(200)
																.style("opacity", 0);
										 });

			 }; //end update



			 					  var data_new;
			 					  var data_new_bar
			 					  var sortedData;
			 					  var favourites = [];
			 					  var retweets =[];
			 					  var arcdata = [];
			 						var FavouriteData;
			 						var RetweetData;
			 						var v;

	function map_draw(month){

		   d3.csv("snakebite.csv", function(error,data){
					if(error) throw error;

					var dataArray = data.filter(function(d) {

											if(isNaN(d.Location_lat)){
														return false;
													}
											return true;
										  })

					  g_map.selectAll("circle").remove();
					  bar_g.selectAll("rect").remove();
					  bar_g.selectAll("text").remove();

					   if(month == "Aggregate"){
									data_new  = dataArray
					   }
					   else{
									data_new  = dataArray.filter(function (d) {return formatDate(parseDate(d.Date)) == month});
					  }

					  var countries = data_new.map(function(d) {return d["Location"];});

					  data_new.forEach( function(d) {
									favourites.push(+d.Favourite_count);
									retweets.push(+d.Retweet_count);
									coordinates.push({"Lat":+d.Location_lat , "Lon": +d.Location_lon});
									tweets.push({"Tweet": d.Text});
								d.Month = formatDate(parseDate(d.Date));
								if(!(d.Timezone_lat == "" || typeof d.Timezone_lat == "undefined" || d.Timezone_lat == null || d.Timezone_lat == "NA" || d.Timezone_lat == " "||
								   d.Location_lat == "" || typeof d.Location_lat == "undefined" || d.Location_lat == null || d.Location_lat == "NA" || d.Timezone_lat == " "||d.Location_lat == ""))
								{
										  arcdata.push({type: "LineString", coordinates:[ [+d.Timezone_lat , +d.Timezone_lon],[+d.Location_lat , +d.Location_lon] ] })
								}

					  });

						var colour_range=d3.scale.linear()
											  .domain([d3.min(favourites), d3.min(favourites) + (d3.max(favourites) -d3.min(favourites))/4, d3.max(favourites) - (d3.max(favourites) -d3.min(favourites))/4,d3.max(favourites)])
											  .range(fav_colors);


          for(var i=0, len=data_new.length; i<len; i++){
                data_new[i].Favourite_count  = parseInt(data_new[i].Favourite_count);
								data_new[i].Retweet_count  = parseInt(data_new[i].Retweet_count);
					}


					/// sorting the data in descending order of favourite count for plotting the users with top 5-10 favorited tweets
					FavouriteData = data_new.sort(function (a, b) {
																	return +a.Favourite_count < +b.Favourite_count ? 1 // if b should come earlier, push a to end
																		 : +a.Favourite_count > +b.Favourite_count ? -1 // if b should come later, push a to begin
																		 : 0;                   // a and b are equal
												});
					//if you want to just keep top three
					FavouriteData =  FavouriteData.filter(function(d,i){
																return i < 7;
														});

					RetweetData = data_new.sort(function (a, b) {
																	return +a.Retweet_count < +b.Retweet_count ?  1 // if b should come earlier, push a to end
																		 : +a.Retweet_count > +b.Retweet_count ? -1 // if b should come later, push a to begin
																		 : 0;                   // a and b are equal
												});
					 RetweetData = RetweetData.filter(function(d,i){
															return i < 7;
													});



					console.log(FavouriteData)
					console.log(RetweetData)


      // this changes the bar chart based on the drop down selector : retweets or favourites

						d3.select("#select-bar").on("change", function(){

												var sect = document.getElementById("select-bar");
												v = sect.options[sect.selectedIndex].value;

												if(v == 'Retweets'){
													barupdate(RetweetData, "Retweets");
												}else{
													barupdate(FavouriteData, "Favourites");
												}
						})

						var sect = document.getElementById("select-bar");
						v = sect.options[sect.selectedIndex].value;
						if(v == 'Retweets'){
							barupdate(RetweetData, "Retweets");
						}else{
							barupdate(FavouriteData, "Favourites");
						}

					  var g_map1  = g_map.selectAll("circle")
										 .data(data_new)
										 .enter()
										 .append("circle")
										 .attr("r", 0);  // set the circle size to 0 initially


					  g_map1.transition()
								  .duration(1000)
								  .attr("cx", function(d){
									//console.log(projection([d.Location_lon, d.Location_lat]));
														return projection([d.Location_lon,d.Location_lat])[0];
														})
									  .attr("cy", function(d){
									//console.log(projection([d.Location_lon, d.Location_lat]));
														return projection([d.Location_lon,d.Location_lat])[1];
														})
									  .attr("fill", function(d){
													return colour_range(+d.Favourite_count + 5);
													})
									  .attr('fill-opacity', 0.3)
								  .attr('r', function(d) {
													return(Math.sqrt(+d.Retweet_count + 5));
												  });

							  /*   .transition()
								  .duration(1000)
								  .attr('stroke-width', 0.5)
								  .attr("r", function(d) {
													return (Math.sqrt(+d.Retweet_count)*3 );
												  })
								  .ease('sine')*/



							g_map1.on("mouseover", function(d) {		/// tooltip on mouseover
								           var point = d3.mouse(this)
													 console.log(point[0] + "px")
													div.transition()
														.duration(200)
														.style("opacity", 0.9);
													div.html("<br/>" + "Tweet: "  + d.Text + "<br/>" + "<br/>" + "User:" + d.User + "<br/>" + "Date: " + d.Date + "<br/>" + "Location: " + d.Location + "<br/>" + "Favourite No.: " + +d.Favourite_count  + "<br/>" +  "Retweet No.: " + +d.Retweet_count)
														.style("left", (d3.event.pageX) + "px")
														.style("top", (d3.event.pageY - 28) + "px")
														.style("width", "25%")
														.style("height", "30%")
														.style("font-size", "15px");
													})
								  .on("mouseout", function(d) {
													div.transition()
													   .duration(200)
													   .style("opacity", 0);
									});



						   /// Add colour legend /////

						  var fav_number;
						  fav_number = [d3.min(favourites),Math.round(d3.min(favourites) + (d3.max(favourites) -d3.min(favourites))/4), Math.round(d3.max(favourites) - (d3.max(favourites) -d3.min(favourites))/4), d3.max(favourites)];

						  svg_legend.selectAll("text").remove();  // removing previous bound DOMS in svg

						  svg_legend.selectAll("text")
									.data(fav_number.reverse())
									.enter()
									.append("text")
									.attr("x", legend_width - legend_width/1.2)
									.attr("y", function(d,i){
														return (1.2*i/(fav_number.length))*100 + 3 +'%';
										  })
									.attr("dy", ".55em")
									.style("font-size", "14px")
									.style("font-weight", "bold")
									.style("color", "white")
									.text(function(d){
											   return (d)
									 });

						   $('#legend-title').text("Favourites");
						   $('#legend-title').css({"font-size": 16});
						   $('#legend-title').css({"font-weight": "bold"});


				  });

			};

		   map_draw("Aggregate");

		   //bar_draw("Retweets")

		  $('#map-title').text("Map of Tweet Locations from July 2017 -May 2018");
		  $('#map-title').css({"font-size": "140%"});
		  $('#map-title').css({"font-weight": "bold"});
			 $('#map-title').css({"font-family": "Roboto"});

		  //var select = document.getElementById("select-bar");
			//console.log(select)

		  //select.onchange = function() {
			//  var selIndex = select.selectedIndex;
			//  var selValue = select.options[selIndex].innerHTML;
		 // }


				 ///// Add slider and values //////
		var months_text = ["Aggregate", "July", "August", "September", "October", "November", "December", "January", "February", "March", "April"]

		var months = ["Aggregate", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
		var count = 0;
		var handle;

		var data_index;


			$('#slider').slider({
					min: 0,
					max: months.length - 1,
					step: 1,
				  step: 1,
				  create: function (event, ui) {
					  $('#selectedMonth').text(months_text[0]);
				  },
				  slide: function (event, ui) {
					  $('#selectedMonth').text(months_text[ui.value]);


					  if (months_text[ui.value] == "Aggregate"){
						$('#map-title').text("Map of Tweet Locations from July 2017 -April 2018");
					  }
					  else{
						$('#map-title').text("Map of Tweet Locations for " + months_text[ui.value]);
					  }

						$('#map-title').css({"font-size": "120%"});
						$('#map-title').css({"font-weight": "bold"});

					  data_index = months[ui.value]

					  map_draw(data_index);

						var sect = document.getElementById("select-bar");
						v = sect.options[sect.selectedIndex].value;

						console.log(v)

						if(v == 'Retweets'){
							barupdate(RetweetData, "Retweets");
						}else{
							barupdate(FavouriteData, "Favourites");
						}

				  }
		 });


	     var val = $('#slider').slider("option", "value");

	     // Clicking play button sets slider moving if other conditions are satisfied

		  d3.select("#play-button")
			 .on("click",function() {
									clearInterval(handle);
									handle = setInterval(function(){
														  if (count == (months.length -1)){
																	count = 0;
														  }
														  else{
															count = count + 1;
														  }

														  $("#slider").slider("value",count);
														  var data_index = months[count]
														  map_draw(data_index);
														  $('#selectedMonth').text(months[count]);
												 }, 2000);
							 });

		// clicking pause button stops the slider handle from moving
		   d3.select("#pause-button")
			 .on("click", function() {
								   clearInterval(handle);

						  });

		   // clicking stop resets the slider to the start point at year 2000.
		  d3.select("#stop-button")
			.on("click",function() {
								  clearInterval(handle);
								  count = 0;
								   $('#slider').slider("value",count);
								   var data_index = months[count]
								   map_draw(data_index);
								   $('#selectedMonth').text(months[count]);
						 });


    });



});

$(document).ready(function()

 {
        var frequency = new Array();

        d3.csv("snakebite.csv", function(error, csv_data) {
                  if (error) throw error;

                  csv_data.forEach(function(d){
                   d.frequency = 1;
                   d.Date = parseInt(d.UnixTime);
                /// the date is converted to milliseconds using the formula in excel: A1-DATE(1970,1,1))*86400
                  });

                  var data =d3.nest()
                    .key(function(d) {
                      return +d.Date;
                    })
                    .sortKeys(d3.ascending)
                    .rollup(function(g) {
                      return {
                        'frequency':d3.sum(g, function(d) {return d.frequency;}),
                      }
                    })
                    .entries(csv_data)

                  for (var i = 0; i < data.length; i++){
                             var value = parseInt(data[i].key)*1000
                             console.log(value)
                            if ( value < 1525132800000){
                                      var item_frequency= new Array(parseInt(data[i].key)*1000, data[i].values.frequency);
                                      frequency.push(item_frequency);
                            }
                  }


                Highcharts.stockChart('tweet-time', {


                        // Create the chart

                       chart: {
                                zoomType: 'xy',
                                backgroundColor: ''
                               },

                      legend: {
                              enabled: true,
                              margin: 10,
                              itemStyle: {
                                 fontSize:'10px',
                                 font: '15pt Trebuchet MS, Verdana, sans-serif',
                                 color: 'black'
                              },
                              itemHoverStyle: {
                                 color: 'green'
                              },
                              itemHiddenStyle: {
                                 color: 'brown'
                              },
                              itemDistance: 25,
                              symbolHeight: 15,
                              symbolWidth: 45,
                              symbolRadius: 35

                        },

                       exporting : {
                                enabled: true
                                 },


                       navigator : {
                                 height: 50,
                                 width: 700,
                                 xAxis: {
                                       labels: {
                                             style: {
                                                  color: '#6D869F',
                                                  fontWeight: 'bold',
                                                  fontSize: "10px"
                                              }
                                         }
                                  },
                                series: [{data: frequency}]

                               },


                        rangeSelector : {

                        buttons: [
                                {
                                  type: 'month',
                                count: 1,
                                text: '1m'
                                },
                                {
                                 type: 'month',
                                count:2,
                                text: '2m'
                                },
                                {
                                 type: 'month',
                                count:3,
                                text: '3m'
                                },
                                {
                                type: 'month',
                                count: 6,
                                text: '6m'
                                },
                                {
                                type: 'month',
                                count: 8,
                                text: '8m'
                                },
                                {
                                type: 'month',
                                count: 10,
                                text: 'All'
                                }
                                ],

                                buttonTheme: {
                                         width: 30,
                                         height: 15,
                                         style: {
                                               color: '#039',
                                               fontWeight: 'bold',
                                               fontSize: "13px"
                                          },
                                },

                               inputBoxBorderColor: 'gray',
                               inputBoxWidth: 70,
                               inputBoxHeight: 20,
                               inputStyle: {
                                        color: '#039',
                                        fontWeight: 'bold',
                                        fontSize: '10px'
                                },
                                labelStyle: {
                                        color: 'black',
                                        fontWeight: 'bold',
                                        fontSize: "10px",
                                         },
                                selected :10,

                              },

                      tooltip: {
                            shadow: false,
                            useHTML: true,
                            style: {
                                    color: 'black',
                                    fontWeight: 'bold',
                                    fontSize: "12px"
                                   }
                             },

                        xAxis:[ {

                               labels: {
                                   style: {
                                            fontSize: '10px'
                                            }
                                      }
                              }],

                        yAxis: [{ // Primary yAxis

                                 gridLineWidth: 0.5,
                                title: {
                                     text: 'Tweet Frequency',
                                     style: {
                                           color: Highcharts.getOptions().colors[1],
                                            fontSize: '14px'
                                           }
                                      },
                                 labels: {
                                         style:{
                                                 fontSize: '12px'
                                               }
                                       },
                                  opposite: false
                                 }],

                        credits:{enabled:false},

                        series: [
                                {
                               name: 'Tweet Frequency',
                               color: 'red',
                               data: frequency,
                               lineWidth: 2,
                               }
                               ],

                });

        });

});

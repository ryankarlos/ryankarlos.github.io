
var network_width = 900,
    network_height = 1000;

var network_svg = d3.select("body").append("svg")
        .attr("width", network_width)
        .attr("height", network_height)
        .append("g")
        .attr("transform",
                "translate(" + 50 + "," + 0 + ")");

var force = d3.layout.force()
        .gravity(0.05)
        .distance(150)
        .charge(-110)
        .size([network_width, network_height]);

  function updatenetwork(year, theme){

        d3.csv("cleaned.csv", function(error, data) {
              if (error) throw error;
              //set up graph in same style as original example but empty
              graph = {"nodes" : [], "links" : []};


        network_svg.selectAll("*").remove()

        data.forEach(function (d) {
    			if (d.Award_Date.slice(-4) == year && d.Theme == theme) {

            console.log(d.Theme)

          graph.nodes.push({ "name": d.Funding_Org });
          graph.nodes.push({ "name": d.Recipient_Org});
          graph.links.push({ "source": d.Funding_Org,
                             "target": d.Recipient_Org,
                             "value": +d.Amount_Awarded });
    												 		}
         });
         // return only the distinct / unique nodes
         graph.nodes = d3.keys(d3.nest()
           .key(function (d) { return d.name; })
           .map(graph.nodes));
         // loop through each link replacing the text with its index from node
         graph.links.forEach(function (d, i) {
           graph.links[i].source = graph.nodes.indexOf(graph.links[i].source);
           graph.links[i].target = graph.nodes.indexOf(graph.links[i].target);
         });
         //now loop through each nodes to make nodes an array of objects
         // rather than an array of strings
         graph.nodes.forEach(function (d, i) {
           graph.nodes[i] = { "name": d };
         });

         force
             .nodes(graph.nodes)
             .links(graph.links)
             .start();

             console.log(graph.nodes)

             // build the arrow.
          network_svg.append("svg:defs").selectAll("marker")
               .data(["end"])      // Different link/path types can be defined here
               .enter().append("svg:marker")    // This section adds in the arrows
               .attr("id", String)
               .attr("viewBox", "0 -5 10 10")
               .attr("refX", 15)
               .attr("refY", -1.5)
               .attr("markerWidth", 6)
               .attr("markerHeight", 6)
               .attr("orient", "auto")
               .append("svg:path")
               .attr("d", "M0,-5L10,0L0,5");

       // add the links and the arrows
         var path = network_svg.append("svg:g").selectAll("path")
                         .data(graph.links)
                         .enter().append("svg:path")
   //    .attr("class", function(d) { return "link " + d.type; })
                        .attr("class", "link")
                       .attr("marker-end", "url(#end)")

                      console.log(graph.links)

         var network_link = network_svg.selectAll(".network_links")
              .data(graph.links)
             .enter().append("line")
              .attr("class", "network_links")

         network_link.append("title")
               .text(function (d) {
                     return d.value;
                 });

         var network_node = network_svg.selectAll(".network_nodes")
              .data(graph.nodes)
              .enter().append("g")

          var circle = network_node.append("circle")
              .attr("class", "node")
              .attr("id", function (d) { return d.name; })
              .attr('r', function(d) { return d.weight + 3})
              .attr('fill', function(d,i){
                 return color(i)
                  })
              .on("mouseover", handleMouseOver)
              .on("mouseout", handleMouseOut)
               //.attr('cx', function(d) { return d.x;})
               //.attr('cy', function(d) { return d.y; })
              .call(force.drag);

         network_node.append("text")
              .attr("font-size","10px")
              .attr("dx", 8)
              .attr("dy", ".55em")
              .text(function(d) { return d.name; });

          network_node.append("title")
                    .text(function (d) {
                          return d.name;
                      });


          force.on("tick", function() {
               path.attr("d", function(d) {
              var dx = d.target.x - d.source.x,
                      dy = d.target.y - d.source.y,
                      dr = Math.sqrt(dx * dx + dy * dy);
              return "M" +
                      d.source.x + "," +
                      d.source.y + "A" +
                      dr + "," + dr + " 0 0,1 " +
                      d.target.x + "," +
                      d.target.y;
          });

              network_node.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
          });

        });

      };

          // Create Event Handlers for mouse
       function handleMouseOver(d, i) {  // Add interactivity
             // Use D3 to select element, change color and size
             d3.select(this).attr({
               fill: "orange",
             });
           }

       function handleMouseOut(d, i) {
             // Use D3 to select element, change color back to normal
             d3.select(this).attr({
               fill: color(i),
             });
           }

       updatenetwork("2017","elderly")

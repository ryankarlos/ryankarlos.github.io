---
title: "Orchestrating ETL workflow using AWS Step Function"
categories: ['aws','step function','glue', 'etl', 'states', 'athena']
tags: ['aws','step function','glue', 'etl', 'states', 'athena']
url: https://github.com/ryankarlos/AWS-ETL-Workflows
comments: true
---
___

AWS Step Functions is a serverless orchestration service that allows users to create workflows as a series of event-driven steps.
Step Functions is based on state machines and tasks. A state machine is a workflow. A task is a state in a workflow 
that represents a single unit of work that another AWS service performs. Each step in a workflow is a state.


{% capture products %}
{% remote_include https://raw.githubusercontent.com/ryankarlos/AWS-ETL-Workflows/master/step_functions/README.md %}
{% endcapture %}

{{ products | replace_first:"# Step Function for Running Glue Job"  }}
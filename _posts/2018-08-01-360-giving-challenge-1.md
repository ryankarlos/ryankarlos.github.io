---
title: "360Giving Data Visualisation Challenge - Part 1"
categories: ['visualisation','tableau']
tags: ['visualisation','tableau','funding', 'challenge']
url: https://github.com/ryankarlos/GrantNav_Tableau
comments: true
---

{% capture products %}
{% remote_include https://raw.githubusercontent.com/ryankarlos/GrantNav_Tableau/master/README.md %}
{% endcapture %}

{% assign title = page.title | prepend: "# "   %}

{{ products | replace_first:title  }}

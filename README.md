 [![Gem Version](https://img.shields.io/badge/gem-v5.2.1-green)](https://rubygems.org/gems/jekyll-theme-chirpy)
 [![Build Status](https://github.com/ryankarlos/ryankarlos.github.io/actions/workflows/pages-deploy.yml/badge.svg?branch=master)](https://github.com/ryankarlos/ryankarlos.github.io/actions?query=branch%3Amaster+event%3Apush)

# Website for Data Science and Cloud Blogs
___

The website containing all my blogs can be accessed [here](https://www.ryannazareth.com/)

This respository uses the [Jekyll Chirpy theme](https://github.com/cotes2020/jekyll-theme-chirpy/blob/master/README.md) 
with slight customisation. 

Gh-actions workflow is configured to build and deploy a Jekyll site to Github Pages.
The site contents are first built and stored in local `_site` dir. The compose action 
[actions/upload-pages-artifact](https://github.com/actions/upload-pages-artifact) then packages and uploads the 
artifact with default `github-pages` name. The [deploy-pages](https://github.com/actions/deploy-pages) action
is then used to deploy the artifact to Github Pages.

.PHONY: serve

serve:
	export LC_ALL="en_US.UTF-8"
	export LANG="en_US.UTF-8"
	export LANGUAGE="en_US.UTF-8"
	bundle update
	bundle exec jekyll serve


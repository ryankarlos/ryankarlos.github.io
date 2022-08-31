---
title: "Application of Natural Language Processing and Visualisation for Tracking Twitter Discussion Around Global Snakebite­­ Disease
"
categories: ['nlp','twitter','dataviz', 'machine learning', 'global health']
tags:  ['nlp','twitter','dataviz', 'machine learning', 'global health']
author: Ryan Nazareth
comments: true
---

In this blog, I will introduce some work I was doing as part of my MSc project in collaboration with a company 
I was working for at the time. The purpose was to develop a tool which could be used by people in the public sector 
(charities, global health organisations) to gain a better understanding of the discussions related to the snakebite 
disease which kills over 100,000 people a year, and more prevalent in poorer parts of the world due to lack of 
unaffordable treatments and anti-toxins. In June 2017, the World Health Organisation (WHO) added it to the 
priority list of neglected tropical diseases. All the code used, in the snippets is available in [github](https://github.com/ryankarlos/Twitter_Analytics)

The data was collected using Twitter’s Standard Search API. The API has strict limitations on the 
quantity of tweets sampled for a given query and is restricted to the last 7 days and is rate limited for a 15-minute. 
The ‘TwitterSearch’ package in Python was used, to make calls to the API. Each search query was a keyword or n-gram 
of words which were thought to be most relevant to the topic of snakebite epidemic. These keywords were initially 
set as ‘#snakebite’, ‘snake anti-venom’, ‘snakebite ntd’, ‘snakebite’, ‘snake venom’. Data was collected at weekly 
intervals for a 6-month period from 3 July 2017 to 3 January 2018 using API calls for each of the search queries. 
The columns of the dataset included ‘the user twitter handle’, ‘the user who received the tweet’, ‘location’, 
‘favourite count’, ‘number of retweets’, ‘whether tweet was retweeted or not’, ‘tweet text’. An additional 
‘sentiment’ column was included to store the manually labelled sentiment values. The total number of tweets 
captured for this period was 1750.

## Analytics

Word clouds of top word frequency were generated after cleaning and tokenising the 
entire corpus of tweets into unigrams and bigrams and excluding the keywords used for 
searching the tweets i.e. ‘snake’, ‘snakebite’, ‘venom’. The top 5 most common 
single words (frequency above 150) were ‘gombe’, ‘n8m’, ‘approves’, ‘venomous’, ‘antivenom’ 
whilst the top 5 bigrams (frequency above 100) were ‘approves n8m’, ‘gombe approves’, ‘gombe state’, 
‘state government’, ‘government approves’. This is related to tweets about the Gombe state in 
Nigeria approving N8 million project funding for research into developing more effective local snake anti-venom, which were heavily mentioned by a number of users. By inspecting the uni gram 
and bigram word clouds further, we can also see other words related to snakebite disease such as 
‘health’, ‘envenoming’, ‘deadly’, ‘treat’ ‘venomous bites’, ‘ntd’, ‘nigeria’ and more specific 
incident related words such as ‘protect owner’, ‘courageous dog’. There is also evidence of 
research themes in the corpus through words such as ‘professor working’, ‘carbon monoxide’, 
‘funding’, ‘project’, due to the large number of tweets related to publications and 
cutting-edge research into snakebite treatment around the world. 

<img src="screenshots/twitter-analytics/bag-of-words.png">

 
The stacked bar chart plotted in Tableau showing thethe top 10 users who accumulated the most favourites and 
retweets (with a lower limit of 30 favourites or 30 retweets) during this period. The users who were accumulating 
the most favourites and retweets came from news organisations or online publishers such as Mashable ,
New York Times, WebMD (@mashable, @nytimes, @WebMD), global health organisations such as the London School 
of Tropical Medicine, Médecins Sans Frontières, Health Action International (@LSTM, @MSFsci, @WebMD, 
@HAImedicines), global health researchers or policy advisors like Nick Casewell, Peter Hotez, Julien 
Potet (@nickcasewell, @PeterHotez, @julienpotet) and the rest from bot or anonymous accounts (@Belxab,
@ALT_uscis). The New York Times was very active during this period and given its global popularity and 
online following, its not suprising that it got almost 750 retweets and favourites during this period. 
Interestingly, the Spanish bot account @Belxab generated the second most number of favourites and retweets 
(276) from only a single tweet “Efecto del veneno de serpiente sobre la sangre” which translates as “The 
effect of snake venom on blood”. 


<img src="screenshots/twitter-analytics/tweet-freq.png">

The global distribution of tweets which were favourited or retweeted at least once for the aggregate 
6-month period of this study is shown in figure 2. New York had the highest number of favourites and 
retweets primarily due to a couple of tweets by the @nytimes (figure 17b) regarding an 
incident involving a Nepali woman who was forced to live in a hut and died of snakebite. Within Australia, 
the most favourites were generated by a tweet from @CSIROnews (figure 17b) which described an 
interesting fact about a snake species: “Brown snake venom seems to shift as they grow, tailoring to a 
changing diet of prey”. Within India, the most prominent tweets were from the South in Bangalore and 
the East (Kolkata) by the Bangalore Mirror(@BangaloreMirror) and the Eastern Command of the Indian 
Army(@easterncomd) respectively. The former tweeted: “Brave act: Brother sucks out snake venom, saves sister” 
describing a local snakebite incident in Karnataka whilst the latter’s tweet (“#Army saved life of snakebite victim 
in Tamenglong, Manipur by providing timely anti-venom treatment.”) was to do with the Indian army 
providing anti- venom in an improverished area (Manipur) to save someone’s life.  In the UK, tweets were 
originating mainly from North West (Liverpool), Oxford, Cambridge and London. These are associated with the high 
concentration of tropical medicine organisations and researchers in these cities. In Nigeria, tweets in Lagos in 
the west and Abuja in central region were mainly from local news accounts. None of the users in the UK and Nigeria 
were in the list of the top 10 favourited users during the entire 6-month period .


<img src="screenshots/twitter-analytics/map.png">


The `topicmodels` package in R programming language provides an interface to the source code for a Latent 
Dirichlet Allocation (LDA) model, a generative probabilistic model in which each document in a text corpus is 
modelled as a random mixture of a set of latent topics. The LDA() function in the topicmodels package requires
the user to specify the number of topics k and is the only parameter which requires input. Based on some 
prior knowledge of the current state of snakebite disease and the topics of importance that people are most 
likely to tweet about (e.g. research, incidents, government funding), it was estimated that the value of k 
could range between 5 and 10. 
So a range of k values from 5 to 10 were tested incrementally and based on the clusters of words generated, 
the value with k = 10 gave the best results with more unique topic separations compared to lower 
values as seen in the table below. Topics 1 and 5 seem to deal with general snakebite death statistics, 
topic 2 deals with NTD treatment programs, topics 8 and 9 deal with specific incidents, topics 4 and 
7 are related to Nigerian government policy and topics 3 and 10 are related to new carbon monoxide research 
treatment and a professor working on getting funding for a project respectively. Topic 6 was more ambiguous due 
to the words ‘australian’, ‘nigerian’, ‘govt’, ‘deaths’, ‘brown’ which suggests that this could be a 
mixture of numerous topic areas. 

|Topic | Top 10 words with highest probabilities  |
|:------:|:------:|
|1 |anti, blood, snakes, know, bitten, envenoming, getting, like, dies, victims |
|2 |people, ntd, programs, treatment, year, million, poor, diseases, learn, reaching |
|3| new, carbon, medicine, monoxide, research, quest, effects, modernize, nigeria, journal | 
|4| gombe, approves, government, state, news antisnake, hope, common, time, worldwide|
|5| antivenom, bite, ministry, senate, provide, victims, wants, died, crisis, due| 
|6| deaths, govt, scarcity, human, Australian, one, brown, nigerian, body, worlds | 
|7| health, nigeria, stock, adewole, minister, lacking, india, says, Isaac, tropical| 
|8| dog, takes, courageous, owner, protect, need, woman, snaksymp, even, live|
|9| snakebites, pit, bit, venomous, bulls, farm, got, children, antivenoms, kids| 
|10| venomous, bites, deadly, working, professor, treating, funding, project, aimed, kill| 





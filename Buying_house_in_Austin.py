#!/usr/bin/env python
# coding: utf-8

# # Best Neighborhood to buy a house

# "Companies like Google and Tesla are moving operations to Austin. The software giant Oracle has also relocated its headquarter here. As more companies move here, that means more people looking for homes, and the city is also attractive to outside investors. With a steady influx of job creation in the pipeline, the housing market will continue to post strong numbers well into 2021.
# 
# Big companies moving here will also play into what happens to the housing market. With historically low mortgage interest rates (below 3%) and an all-time high in corporate relocations, the housing demand is way up and the supply side cannot match up."
# 
# This is from a report on Austin's real estate market (https://www.noradarealestate.com/blog/austin-real-estate-market/)With the house price rocketing in Austin, as well as the historically low mortgage rate, more and more people living nearby consider to buy a house in Aunstin as an investment. My husband and I are one of them.
# 
# In order to evaluate the neighborhoods in Austin from the above three different aspects, we will need to utilize the data from three resources: 1) Foursquare for the venues in each neighborhoods in Austin; 2) crime rate data of each neighborhood in Austin; 3) house price statistics of each neighborhood in Austin.
# 1. Venues data from Foursquare 
# Using Foursquare API, we can request all the venues in all neighborhoods in Austin. The neighborhoods can be further clustered according to the most popular venue types, which will provide an indicator as to which neighborhood is convenient for specific needs
# 2. Crime data
# Crime data of each neighborhoods in Austin can be found from austintexas.com. The search will be limited to recent one year (from March 1st 2020 to March 1st 2021) and the list will be scraped using beautifulsoup library and key information including address and crime type will be scraped to form a table. The final statistics will be compared with the report from NeighborhoodScout.com.
# 3. House price data
# The average or median house price by neighborhood in Austin will be obtained from the report in Texas Real Estate Research Center (https://www.recenter.tamu.edu/data/housing-activity/#!/activity/MSA/Austin-Round_Rock) and compared with the report in NeighborhoodScout website (https://www.neighborhoodscout.com/tx/austin/real-estate).
# 

# ## 1. Clustering neighborhoods in Austin using Foursquare API

# Before we get the data and start exploring it, let's download all the dependencies that we will need.

# In[77]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

#import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().run_line_magic('pip', 'install folium')
import folium!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# I was trying to get a full list of the neighborhood of Austin and hopefully also their coordinates. But there seems not such clean data in a table format to use. So we also need to Import bs4 and request so that we can scrape information from needed websit.

# The websit has a list of neighborhoods in Austin--https://en.wikipedia.org/wiki/List_of_Austin_neighborhoods We can scrape the list and save it into a dataframe, then use geocoder to get their coordinate.

# In[6]:


get_ipython().system('pip install geopy')
get_ipython().system('pip install bs4')
#!pip install requests
from bs4 import BeautifulSoup # this module helps in web scrapping.
import requests  # this module helps us to download a web page


# In[13]:



url='https://en.wikipedia.org/wiki/List_of_Austin_neighborhoods'
data=requests.get(url).text
soup=BeautifulSoup(data,"html5lib")
href_list=soup.find_all(href=True)
href_list


# We can see from the list that not only the neighborhood names are scraped. So we need to limit what can be written into our target dataframe. After some examination, we can tell that 'Austin, Texas' ends every neighborhood name, which can be used as the critera. And it works great! Now we have the list of neighborhood names in a dataframe. 
# 

# In[147]:


import pandas as pd
neigh_df = pd.DataFrame(columns=["Neighborhood"])

for i,href in enumerate(href_list):
    neigh=href.get('title')
    if (neigh!=None) and (", Austin, Texas" in neigh):        
        neigh_df=neigh_df.append({"Neighborhood":neigh}, ignore_index=True)
    
neigh_df


# In[148]:


neigh_df['Neighborhood']=neigh_df['Neighborhood'].str.replace(r"\(.*\)","")
neigh_df.drop_duplicates(inplace=True)
len(neigh_df)


# We can see after deleting the duplicated rows we have 64 neighorhoods in the dataframe.

# In[151]:


get_ipython().run_line_magic('pip', "install geopy # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
    


# In[152]:


neigh_df['Latitude']=0
neigh_df['Longitude']=0


for i in neigh_df.index:
          
    geolocator = Nominatim(user_agent="ny_explorer")
    location = geolocator.geocode(neigh_df.loc[i,'Neighborhood'])
    if location!=None:
        neigh_df.loc[i,'Latitude'] = location.latitude
        neigh_df.loc[i,'Longitude'] = location.longitude
    else:
        print("coord not found for",neigh_df.loc[i,'Neighborhood'] )


# So there are three neighborhood of which the coordinates cannot be found. We searched for their zip code and try to find their coordinate by zip code--
# * Spyglass-Barton's Bluff, Austin, Texas--TX 78746
# * The Ridge at Lantana, Austin, Texas --TX 78735
# * North Burnet–Gateway, Austin, Texas--TX 78757

# In[153]:


zip_dic={"Spyglass-Barton's Bluff, Austin, Texas":"TX 78746","The Ridge at Lantana, Austin, Texas ":"TX 78735","North Burnet–Gateway, Austin, Texas":"TX 78757"}

for neigh,code in zip_dic.items():
    geolocator = Nominatim(user_agent="ny_explorer")
    location = geolocator.geocode(code)
    for i in neigh_df.index:
        if neigh_df.loc[i,'Neighborhood']==neigh:
            neigh_df.loc[i,'Latitude'] = location.latitude
            neigh_df.loc[i,'Longitude'] = location.longitude
            print(neigh, 'is added with', location.latitude, location.longitude)


# Great! Now we have 64 neighborhood and they all have latitude and longitude ready for further searching! Let's reset the index to have the index reflect the number of items in the table.

# In[154]:


neigh_df.reset_index(drop=True,inplace=True)


# In[155]:


neigh_df.tail(30)


# Now we are ready to aquire the venue information using Foursquare API!

# In[156]:


CLIENT_ID = 'KTH12FDY3PFVT1TVVGICGZ1EAG0VI3EPPDG0BXU1NER0R232' # your Foursquare ID
CLIENT_SECRET = 'FWNXW4CB3EHEQQFUJN0CLRPNUIUIQEWIGCJHSYWQRGG1JNMX' # your Foursquare Secret
ACCESS_TOKEN = 'AFYA3MZXBJ0MAQ5JFQLJ44GEGZAAI0R5COVXFFKEIFYN2O4M' # your FourSquare Access Token

VERSION = '20180605' # Foursquare API version
LIMIT = 100 # A default Foursquare API limit value

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# In[157]:


# define function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# In[191]:


url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            30.290208, 
            -97.747384, 
            1000, 
            LIMIT)
            
        # make the GET request
results = requests.get(url).json()["response"]['groups'][0]['items']


# In[197]:


results[3]['venue']


# In[158]:


def getNearbyVenues(names, latitudes, longitudes, radius=1000):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[159]:


austin_venues = getNearbyVenues(names=neigh_df['Neighborhood'],
                                   latitudes=neigh_df['Latitude'],
                                   longitudes=neigh_df['Longitude']
                                  )


# Let's check how many venues we have collected in total.

# In[160]:


austin_venues


# In[161]:


import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
austin_venues.groupby('Neighborhood').count()


# we can see that there are 11 neighborhoods which have only aquired less than 15 venues. The least number of venues is 3 which is of Spyglass-Barton's Bluff;and The Ridge at Lantana only has 4. Checking the map, we can see that these neighborhoods are far away from Austin city. So let's exclude these 11 neighborhoods from our list.

# In[162]:


venue_group=austin_venues.groupby('Neighborhood').count()
#create list of neighborhoods to drop
drop_list=venue_group[venue_group['Venue']<15].index.tolist()
for i in drop_list:
    austin_venues.drop(austin_venues[austin_venues['Neighborhood']==i].index, inplace = True) 


# We can also double check if these neighborhoods have been deleted from the list.

# In[163]:


for i in drop_list:
    
    print(i in austin_venues.Neighborhood.tolist())


# In[164]:


print('There are {} uniques categories.'.format(len(austin_venues['Venue Category'].unique())))


# In[165]:


num_top_venues = 5

for hood in austin_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = austin_grouped[austin_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[166]:


# one hot encoding
austin_onehot = pd.get_dummies(austin_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
austin_onehot['Neighborhood'] = austin_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [austin_onehot.columns[-1]] + list(austin_onehot.columns[:-1])
austin_onehot = austin_onehot[fixed_columns]

austin_onehot.head()


# In[167]:


austin_grouped = austin_onehot.groupby('Neighborhood').mean().reset_index()
austin_grouped


# In[168]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[169]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = austin_grouped['Neighborhood']

for ind in np.arange(austin_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(austin_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# In[177]:


# set number of clusters
kclusters = 6

austin_grouped_clustering = austin_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(austin_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[178]:


# add clustering labels
del neighborhoods_venues_sorted['Cluster Labels']
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
#delete the 6 neighborhoods from the neigh_df as well
for i in drop_list:
    neigh_df.drop(neigh_df[neigh_df['Neighborhood']==i].index, inplace = True) 


austin_merged = neigh_df

# merge data to add latitude/longitude for each neighborhood
austin_merged = austin_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

austin_merged.head() 


# ## 2. Analyze the clusters and find our target
# Let's first show the locations of all the neighborhoods and their cluster labels in map using folium.

# In[179]:


# create map30.2672° N, 97.7431° W
map_clusters = folium.Map(location=[30.2672, -97.7431], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(austin_merged['Latitude'], austin_merged['Longitude'], austin_merged['Neighborhood'], austin_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# We can see cluster 3 and 4 are close to city center, while 2 and 5 are dominated and scattered everywhere. Now let's try to analyze the characteristics of these 6 clusters. First we need to narrow the venue catergories.
# Let's also mark the few neighborhoods that are close to the river, which will be the good spots! They are Northwest Hills,  Tarrytown, Bryker Woods, West Campus, Oak Hill.

# In[187]:


austin_venues.groupby('Venue Category').count().sort_values(by='Venue',ascending=False)


# It's interesting to find that there are many mexican restaurants in Austin! And no price that coffee shot and Park are also everywhere.Other than the most popular venus (top 10), I personally care the most if I am close to a bakery, ice cream shop, and also yoga studio and spa. I am also pretty sure they mean the most for people of similar age as me! So the my customized category list would be including Coffee Shop, Park, Bar, Grocery Store, Bakery, Ice Cream Shop, Yoga Studio,Spa. Let's also combine all types of restaurants into one type and add it to our list and see how much the different clusters are dominated by these venues I am most interested in!

# In[207]:


res_list=['Neighborhood']
for i in austin_grouped.columns:
    if 'Restaurant' in i:
        res_list.append(i)
        
res_list


# Wow we have included a variety of restanrants into the list.

# In[219]:



venue_res=austin_grouped[res_list].sum(axis=1)
austin_category_group=austin_grouped.copy()

del austin_category_group['Restaurant']
austin_category_group.insert(0, 'Restaurant', venue_res.tolist())
austin_category_group.insert(0, 'Cluster Labels', kmeans.labels_)



cate_list=['Neighborhood','Cluster Labels','Restaurant','Coffee Shop', 'Park', 'Bar', 'Grocery Store', 'Bakery', 'Ice Cream Shop', 'Yoga Studio','Spa']
austin_category_group=austin_category_group[cate_list]
austin_category_group


# In[224]:


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
austin_label_df=austin_merged.copy()
fig, axes = plt.subplots(1,kclusters, figsize=(20, 5), sharey=True)

axes[0].set_ylabel('Count of venues (relative)', fontsize=25)
#plt.set_xlabel('Venue category', fontsize='x-large')

for k in range(kclusters):
    #Set same y axis limits
    axes[k].set_ylim(0,0.3)
    axes[k].xaxis.set_label_position('top')
    axes[k].set_xlabel('Cluster ' + str(k), fontsize=25)
    axes[k].tick_params(labelsize=20)
    plt.sca(axes[k])
    plt.xticks(rotation='vertical')
    sns.boxplot(data = austin_category_group[austin_category_group['Cluster Labels'] == k].drop('Cluster Labels',1), ax=axes[k])

plt.show()


# Since all these categories mean a lot to us so we can easily get rid of the clusters of which certain category is missing. So cluster 0 and 1, which miss quite a few, cluster 3, which don't have many bakery while is populated by bars, are no longer interesting to us. Cluster 4 lacks yoga studio, which certainly decreases its attractiveness as well. What is left now would be cluster 2 and 5. Checking back in the map, they are scattered around the city while centered around the city center. We still have a lot to choose from! (We don't like to hang out in bars very much, and have the impression that too many bars many to some extent indicate unsafety.But we know many young people like to go to bars after work, so I decide to include cluster 5 for your interest as well.)

# In[227]:


austin_merged.head()


# In[228]:


#list cluster 2 and 5 and their most popular venue types
austin_merged.loc[austin_merged['Cluster Labels'] == 2, austin_merged.columns[[0] + list(range(4, austin_merged.shape[1]))]]


# It impressed me that Barrington Oaks and Highland both have a lot of assian food. And Westgate has Szechuan restaurants as the 8th most common venue whil West Campus has Bubble Tea shop as the 7th most common. I like them! 

# In[229]:


#list cluster 2 and 5 and their most popular venue types
austin_merged.loc[austin_merged['Cluster Labels'] == 5, austin_merged.columns[[0] + list(range(4, austin_merged.shape[1]))]]


# After checking the common venue lists of the cluster 5, I decide that they are much less attractive to me compared to cluster 2, which have a lot of assian food, sweet shops, as well as parks and yoga studios. So I am going to limit my selection to cluster 2.

# ## 3. Compare house prices
# Let's get the data using beautifulsoup again from http://www.city-data.com/nbmaps/neigh-Austin-Texas.html#N121

# In[231]:


url2='http://www.city-data.com/nbmaps/neigh-Austin-Texas.html#N121'
data=requests.get(url2).text
soup=BeautifulSoup(data,"html5lib")
tables=soup.find_all('table')
tables


# In[242]:


cluster2=austin_merged.loc[austin_merged['Cluster Labels'] == 2, austin_merged.columns[[0] + list(range(4, austin_merged.shape[1]))]]['Neighborhood'].tolist()
cluster2_list=[]
for i in cluster2:
    cluster2_list.append(i.split(',', 1)[0])
    
cluster2_list


# Let's try to loop over the tables to see if the information we are looking for can be scraped.

# In[246]:


price_data = pd.DataFrame(columns=["Neighborhood", "Median income"])
for neig in cluster2_list:
    
    for i in range(len(tables)):
    
        table=pd.read_html(str(tables[i]), flavor='bs4')
        neighborhood = np.array(table[0])[0][0]
        price = np.array(table[0])[0][1]
        if (neighborhood==neig) and ('$' in price):
            price_data = price_data.append({"Neighborhood":neighborhood, "Median income":price}, ignore_index=True)
       

price_data.tail()


# The scraping loop is not able to collect the desired information. After checking the website in detail, we can see that inside each neighborhood, different types of information are provided, which can make the flow complicated. Since we have narrowed into just a few neighorhood. Note that since we also want to live close to the river, the target neighborhoods then are Northwest Hills,  Tarrytown, Bryker Woods, West Campus.Let's find them by searching the websit and compare. 

# In[252]:


price_data = pd.DataFrame(columns=["Neighborhood", "Median income","Ave house price"])

price_data=price_data.append({"Neighborhood":"Northwest Hills", "Median income":98332,"Ave house price":480685},ignore_index=True)
price_data=price_data.append({"Neighborhood":"Tarrytown", "Median income":103875,"Ave house price":855988},ignore_index=True)
price_data=price_data.append({"Neighborhood":"Bryker Woods", "Median income":83931,"Ave house price":736292},ignore_index=True)
price_data=price_data.append({"Neighborhood":"West Campus", "Median income":22642,"Ave house price":1423994},ignore_index=True)



price_data.sort_values(by=['Ave house price','Median income'])


# In[254]:


price_data.plot.bar(x='Neighborhood')


# It can be seen that for Northwest, Tarrytown, and Bryker Woods,the average house prices are proportional with median house income. This is pretty normal.For West campus,most houses are for renting to students at university, which explains why the median income is low while house price is so high. While it's a good place for investment, the price is too high, plus it's not economic to live there ourselves. Let's consider it a good investment opportunity when we get richer. And with the same reason, Northwest Hills are most affordable for us! Lastly I checked the crime rate, it's also relatively safe compared to other neighborhoods near city center. That's our final target area--Northwest Hills!

# In[ ]:





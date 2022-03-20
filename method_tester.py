#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import HTML
import numpy as np
plt.ioff();


# In[8]:


DATA_URL = 'https://raw.githubusercontent.com/splotniks/inequal4/main/GCIPrawdata.csv'


EXPLANATION = ""
HTML("""<style>
.app-subtitle {
    font-size: 1.5em;
}

.app-subtitle a {
    color: #106ba3;
}

.app-subtitle a:hover {
    text-decoration: underline;
}

.app-sidebar p {
    margin-bottom: 1em;
    line-height: 1.7;
}

.app-sidebar a {
    color: #106ba3;
}

.app-sidebar a:hover {
    text-decoration: underline;
}
</style>
""")


# In[3]:


def GiniArea(H): 
    H = np.sort(H)
    H_total = sum(H)
    cumul_H = np.cumsum(H).tolist() 
    portions_H=[0]
    for i in cumul_H:
        portions_H.append(cumul_H[cumul_H.index(i)]/H_total*100) 
    B = np.trapz(portions_H, dx = 1/len(H))
    GiniApprox = ((50-B)/50)

    return (GiniApprox)


# In[4]:


df2 = pd.read_csv(DATA_URL)
goodC = np.where((df2.groupby('Country').count()['Year'] !=35),0,df2.groupby('Country').count().index)
df2 = df2[df2['Country'].isin(goodC)]
X = []
for x in range (0,len(df2)):
    X.append(GiniArea(np.array(df2.iloc[x, 2:12]).astype(np.double)))
df2['Gini'] = X
df2['Income ratio 50:10'] = df2['Decile 5 Income']/df2['Decile 1 Income']
df2['Income ratio 90:50'] = df2['Decile 9 Income']/df2['Decile 5 Income']
df2['Income ratio 90:10'] = df2['Decile 9 Income']/df2['Decile 1 Income']


# In[6]:


class App1:
#version1
    
    def __init__(self, df):
        self._df = df
        available_indicators = self._df['Country'].unique()
        available_metrics = ['Income ratio 50:10', 'Income ratio 90:50', 'Income ratio 90:10', 'Gini', 'All']
        self._x_dropdown = self._create_indicator_dropdown(available_indicators, -7)
        self._y_dropdown = self._create_indicator_dropdown(available_metrics, 3)  
        self._plot_container = widgets.Output()
        self._year_slider, year_slider_box = self._create_year_slider(
            min(df['Year']), max(df['Year'])
        )

        _app_container = widgets.VBox([
            widgets.HBox([self._x_dropdown, self._y_dropdown]),
            self._plot_container,
            year_slider_box
        ], layout=widgets.Layout(align_items='center', flex='3 0 auto'))
        self.container = widgets.VBox([
            widgets.HTML(
                (
                    """<h1>Simple Inequality Measures</h1>
                    <p><em>Comparisons within a country</em><p>

<p>The 90:10 ratio measures the 9th decile of income over the 1st. The larger this number is, the more unequal the income share between these groups is.

</p> The Gini will be discussed later on, but provides an equality measure that factors in all income deciles.
"""          
                ), 
                layout=widgets.Layout(margin='0 0 5em 0')
            ),
            widgets.HBox([
                _app_container, 
                widgets.HTML(EXPLANATION, layout=widgets.Layout(margin='0 0 0 2em'))
            ])
        ], layout=widgets.Layout(flex='1 1 auto', margin='0 auto 0 auto', max_width='1024px'))
        self._update_app()     
        
    @classmethod
    def from_url(cls, url):
        df = pd.read_csv(url)
        #df['Gini'] = GiniArea(df.iloc[:, 2:12])
        X = []
        for x in range (0,len(df)):
            X.append(GiniArea(np.array(df.iloc[x, 2:12]).astype(np.double)))
        df['Gini'] = X
        #df['Gini'] = GiniArea(np.array(df.iloc[1:, 2:12]).astype(np.double))
        df['Income ratio 50:10'] = df['Decile 5 Income']/df['Decile 1 Income']
        df['Income ratio 90:50'] = df['Decile 9 Income']/df['Decile 5 Income']
        df['Income ratio 90:10'] = df['Decile 9 Income']/df['Decile 1 Income']
        return cls(df)
        
    def _create_indicator_dropdown(self, indicators, initial_index):
        dropdown = widgets.Dropdown(options=indicators, value=indicators[initial_index])
        dropdown.observe(self._on_change, names=['value'])
        return dropdown
    
    def _create_year_slider(self, min_year, max_year):
        year_slider_label = widgets.Label('Year range: ')
        year_slider = widgets.IntRangeSlider(
            min=min_year, max=max_year,
            layout=widgets.Layout(width='500px')
        )
        year_slider.observe(self._on_change, names=['value'])
        year_slider_box = widgets.HBox([year_slider_label, year_slider])
        return year_slider, year_slider_box
    
    def _create_plot(self, x_indicator, y_indicator, year_range):
        df = self._df[self._df['Year'].between(*year_range)]
        plt.figure(figsize=(10, 7))
        years = np.arange(year_range[0], year_range[1]+1,1)
        #plt.gca().tick_params(axis='both', which='major', labelsize=16)
        if y_indicator =='All':
            x1 = df[df['Country'] == x_indicator]['Income ratio 50:10']
            x2 = df[df['Country'] == x_indicator]['Income ratio 90:50']
            x3 = df[df['Country'] == x_indicator]['Income ratio 90:10']
            x4 = df[df['Country'] == x_indicator]['Gini']
            fig, ax1 = plt.subplots(figsize=(10, 7)) 
            plot_1 = ax1.plot(years, x1, label = 'Income ratio 50:10') 
            ax1.plot(years, x2, label = 'Income ratio 90:50' )
            ax1.plot(years, x3, label = 'Income ratio 90:10' )
            ax1.set_xlabel(x_indicator, size=16)
            ax1.set_ylabel("Income ratios", size=16)
            ax1.legend(loc='center right')
            ax2 = ax1.twinx() 
            ax2.set_ylabel('Gini',size=16)
            plot_2 = ax2.plot(years, x4, label = 'Gini', color='black' )
            ax2.legend(loc='upper right')
            
        else:
            xs = df[df['Country'] == x_indicator][y_indicator]
            plt.xlabel(x_indicator, size=16)
            plt.ylabel(y_indicator, size=16)
            plt.plot(years, xs)
        
    def _on_change(self, _):
        self._update_app()
        
    def _update_app(self):
        x_indicator = self._x_dropdown.value
        y_indicator = self._y_dropdown.value
        year_range = self._year_slider.value
        self._plot_container.clear_output(wait=True)
        with self._plot_container:
            self._create_plot(x_indicator, y_indicator, year_range)
            plt.show()


# In[9]:


app = App1.from_url(DATA_URL)

app.container


# In[10]:


class App2:
#version2
    
    def __init__(self, df):
        self._df = df
        available_indicators = self._df['Country'].unique()
        available_metrics = ['Income ratio 50:10', 'Income ratio 90:50', 'Income ratio 90:10', 'Gini']
        self._x_dropdown = self._create_indicator_dropdown(available_indicators, -7)
        self._y_dropdown = self._create_indicator_dropdown(available_indicators, -8)  
        self._a_dropdown = self._create_indicator_dropdown(available_metrics, 3) 
        self._plot_container = widgets.Output()
        self._year_slider, year_slider_box = self._create_year_slider(
            min(df['Year']), max(df['Year'])
        )

        _app_container = widgets.VBox([widgets.HBox([self._a_dropdown]),
            widgets.HBox([self._x_dropdown, self._y_dropdown]),
            self._plot_container,
            year_slider_box
        ], layout=widgets.Layout(align_items='center', flex='3 0 auto'))
        self.container = widgets.VBox([
            widgets.HTML(
                (
                    """<h1>Simple Inequality Measure</h1>
                    <p><em>Comparisons between 2 countries</em><p>

<p>The 90:10 ratio measures the 9th decile of income over the 1st. The larger this number is, the more unequal the income share between these groups is.

The Gini will be discussed later on, but provides an equality measure that factors in all income deciles."""          
                ), 
                layout=widgets.Layout(margin='0 0 4em 0')
            ),
            widgets.HBox([
                _app_container
            ])
        ], layout=widgets.Layout(flex='1 1 auto', margin='0 auto 0 auto', max_width='1024px'))
        self._update_app()     
        
    @classmethod
    def from_url(cls, url):
        df = pd.read_csv(url)
        X = []
        for x in range (0,len(df)):
            X.append(GiniArea(np.array(df.iloc[x, 2:12]).astype(np.double)))
        df['Gini'] = X
        df['Income ratio 50:10'] = df['Decile 5 Income']/df['Decile 1 Income']
        df['Income ratio 90:50'] = df['Decile 9 Income']/df['Decile 5 Income']
        df['Income ratio 90:10'] = df['Decile 9 Income']/df['Decile 1 Income']
        return cls(df)
        
    def _create_indicator_dropdown(self, indicators, initial_index):
        dropdown = widgets.Dropdown(options=indicators, value=indicators[initial_index])
        dropdown.observe(self._on_change, names=['value'])
        return dropdown
    
    def _create_year_slider(self, min_year, max_year):
        year_slider_label = widgets.Label('Year range: ')
        year_slider = widgets.IntRangeSlider(
            min=min_year, max=max_year,
            layout=widgets.Layout(width='500px')
        )
        year_slider.observe(self._on_change, names=['value'])
        year_slider_box = widgets.HBox([year_slider_label, year_slider])
        return year_slider, year_slider_box
    
    def _create_plot(self, a_indicator, x_indicator, y_indicator, year_range):
        df = self._df[self._df['Year'].between(*year_range)]
        xs = df[df['Country'] == x_indicator][a_indicator]
        years = np.arange(year_range[0], year_range[1]+1,1)
        ys = df[df['Country'] == y_indicator][a_indicator]
        #ys = df[df['Country'] == y_indicator]['Decile 1 Income']
        plt.figure(figsize=(10, 7))
        plt.xlabel(x_indicator, size=16)
        plt.ylabel(y_indicator, size=16)
        plt.gca().tick_params(axis='both', which='major', labelsize=16)
        plt.plot(years, xs, label = x_indicator)
        plt.plot(years, ys, label = y_indicator)
        plt.legend()
        
    def _on_change(self, _):
        self._update_app()
        
    def _update_app(self):
        x_indicator = self._x_dropdown.value
        y_indicator = self._y_dropdown.value
        a_indicator = self._a_dropdown.value
        year_range = self._year_slider.value
        self._plot_container.clear_output(wait=True)
        with self._plot_container:
            self._create_plot(a_indicator,x_indicator, y_indicator, year_range)
            plt.show()


# In[11]:


app2 = App2.from_url(DATA_URL)
app2.container


# In[12]:


class App3:
#version2
    
    def __init__(self, df):
        self._df = df
        available_indicators = self._df['Country'].unique()
        available_metrics = ['Your income ratio', 'Comparison to Gini']
        available_deciles = ['Decile 1 Income','Decile 2 Income','Decile 3 Income','Decile 4 Income','Decile 5 Income','Decile 6 Income','Decile 7 Income','Decile 8 Income','Decile 9 Income','Decile 10 Income'] 
        self._x_dropdown = self._create_indicator_dropdown(available_indicators, -7)  
        self._y_dropdown = self._create_indicator_dropdown(available_metrics, -1) 
        self._a_dropdown = self._create_indicator_dropdown(available_deciles, -1)
        self._b_dropdown = self._create_indicator_dropdown(available_deciles, 0)
        self._plot_container = widgets.Output()
        self._year_slider, year_slider_box = self._create_year_slider(
            min(df['Year']), max(df['Year'])
        )

        _app_container = widgets.VBox([widgets.HBox([self._a_dropdown,self._b_dropdown]),
            widgets.HBox([self._x_dropdown, self._y_dropdown]),
            self._plot_container,
            year_slider_box
        ], layout=widgets.Layout(align_items='center', flex='3 0 auto'))
        self.container = widgets.VBox([
            widgets.HTML(
                (
                    """<h1>Your Own Inequality Measure</h1>
                    <p><em>Select two different income deciles for one country</em><p>
                    
                    

<p>The metric will take your first income decile and divide it by your second.
See which combination will get closest to the Gini for a given country.
"""          
                ), 
                layout=widgets.Layout(margin='0 0 4em 0')
            ),
            widgets.HBox([
                _app_container
            ])
        ], layout=widgets.Layout(flex='1 1 auto', margin='0 auto 0 auto', max_width='1024px'))
        self._update_app()     
        
    @classmethod
    def from_url(cls, url):
        df = pd.read_csv(url)
        X = []
        for x in range (0,len(df)):
            X.append(GiniArea(np.array(df.iloc[x, 2:12]).astype(np.double)))
        df['Gini'] = X
        df['Income ratio 50:10'] = df['Decile 5 Income']/df['Decile 1 Income']
        df['Income ratio 90:50'] = df['Decile 9 Income']/df['Decile 5 Income']
        df['Income ratio 90:10'] = df['Decile 9 Income']/df['Decile 1 Income']
        return cls(df)
        
    def _create_indicator_dropdown(self, indicators, initial_index):
        dropdown = widgets.Dropdown(options=indicators, value=indicators[initial_index])
        dropdown.observe(self._on_change, names=['value'])
        return dropdown
    
    def _create_year_slider(self, min_year, max_year):
        year_slider_label = widgets.Label('Year range: ')
        year_slider = widgets.IntRangeSlider(
            min=min_year, max=max_year,
            layout=widgets.Layout(width='500px')
        )
        year_slider.observe(self._on_change, names=['value'])
        year_slider_box = widgets.HBox([year_slider_label, year_slider])
        return year_slider, year_slider_box
    
    def _create_plot(self, a_indicator, b_indicator, x_indicator, y_indicator, year_range):
        df = self._df[self._df['Year'].between(*year_range)]
        plt.figure(figsize=(10, 7))
        years = np.arange(year_range[0], year_range[1]+1,1)
        if y_indicator =='Comparison to Gini':
            x1 = df[df['Country'] == x_indicator][a_indicator]/df[df['Country'] == x_indicator][b_indicator]
            x4 = df[df['Country'] == x_indicator]['Gini']
            fig, ax1 = plt.subplots(figsize=(10, 7)) 
            plot_1 = ax1.plot(years, x1, label = 'Your income Ratio') 
            ax1.set_xlabel(x_indicator, size=16)
            ax1.set_ylabel("Income ratio", size=16)
            ax1.legend(loc='lower right')
            ax2 = ax1.twinx() 
            ax2.set_ylabel('Gini',size=16)
            plot_2 = ax2.plot(years, x4, label = 'Gini', color='black' )
            ax2.legend(loc='upper right')
            
        else:
            xs = df[df['Country'] == x_indicator][a_indicator]/df[df['Country'] == x_indicator][b_indicator]
            plt.xlabel(x_indicator, size=16)
            plt.ylabel(y_indicator, size=16)
            plt.plot(years, xs)
        
    def _on_change(self, _):
        self._update_app()
        
    def _update_app(self):
        x_indicator = self._x_dropdown.value
        y_indicator = self._y_dropdown.value
        a_indicator = self._a_dropdown.value
        b_indicator = self._b_dropdown.value
        year_range = self._year_slider.value
        self._plot_container.clear_output(wait=True)
        with self._plot_container:
            self._create_plot(a_indicator,b_indicator,x_indicator, y_indicator, year_range)
            plt.show()


# In[13]:


app3 = App3.from_url(DATA_URL)

app3.container


# In[14]:


class App4:
#version2
    
    def __init__(self, df):
        self._df = df
        numbers = [1,2,3,4,5,6,7,8,9,10]
        self._num_dropdown = self._create_indicator_dropdown(numbers, 0) 

        self._app_container = widgets.VBox([widgets.VBox([self._num_dropdown]),
        ], layout=widgets.Layout(align_items='center', flex='3 0 auto'))
        self.container = widgets.VBox([
            widgets.HTML(
                (
                    """Select the number of societies you would like to compare
                    <p><em></em><p>
"""          
                ), 
                layout=widgets.Layout(margin='0 0 4em 0')
            ),
            widgets.HBox([
                self._app_container
            ])
        ], layout=widgets.Layout(flex='1 1 auto', margin='0 auto 0 auto', max_width='1024px'))
        self._update_app()     
        
    @classmethod
    def from_url(cls, url):
        df = pd.read_csv(url)
        return cls(df)
        
    def _create_indicator_dropdown(self, indicators, initial_index):
        dropdown = widgets.Dropdown(options=indicators, value=indicators[initial_index])
        dropdown.observe(self._on_change, names=['value'])
        return dropdown
        
    def _on_change(self, _):
        self._update_app()
        
    def _update_app(self):
        self._num = self._num_dropdown.value


# In[29]:


class App5:
#version2
    
    def __init__(self, df,n):
        self._df = df
        self.num = n
        available_indicators = self._df['Country'].unique()
        self._year_slider, self.year_slider_box = self._create_year_slider(min(df['Year']), max(df['Year']))

        self._dropdowns = self._create_indicator_dropdown(available_indicators, -7 ,self.num)

        self._plot_container = widgets.Output()
        
        self._app_container = widgets.VBox([widgets.VBox(self._dropdowns), 
            self._plot_container,
            self.year_slider_box
        ], layout=widgets.Layout(align_items='center', flex='3 0 auto'))
        self.container = widgets.VBox([
            widgets.HTML(
                (
                    """<h1>Compare multiple countries</h1>
                    <p><em></em><p>
"""          
                ), 
                layout=widgets.Layout(margin='0 0 4em 0')
            ),
            widgets.HBox([
                self._app_container
            ])
        ], layout=widgets.Layout(flex='1 1 auto', margin='0 auto 0 auto', max_width='1024px'))
        self._update_app()     
        
    @classmethod
    def from_url(cls, url, n):
        df = pd.read_csv(url)
        goodC = np.where((df.groupby('Country').count()['Year'] !=35),0,df.groupby('Country').count().index)
        df = df[df['Country'].isin(goodC)]
        X = []
        for x in range (0,len(df)):
            X.append(GiniArea(np.array(df.iloc[x, 2:12]).astype(np.double)))
        df['Gini'] = X
        df['Income ratio 50:10'] = df['Decile 5 Income']/df['Decile 1 Income']
        df['Income ratio 90:50'] = df['Decile 9 Income']/df['Decile 5 Income']
        df['Income ratio 90:10'] = df['Decile 9 Income']/df['Decile 1 Income']
        return cls(df,n)
        
    def _create_indicator_dropdown(self, indicators, initial_index, n):
        dropdown = []
        for x in range (0,n):
            dropdown.append(widgets.Dropdown(options=indicators, value=indicators[initial_index-x]))
            dropdown[x].observe(self._on_change, names=['value'])
        return dropdown      
    
    def _create_year_slider(self, min_year, max_year):
        year_slider_label = widgets.Label('Year range: ')
        year_slider = widgets.IntRangeSlider(
            min=min_year, max=max_year,
            layout=widgets.Layout(width='500px')
        )
        year_slider.observe(self._on_change, names=['value'])
        year_slider_box = widgets.HBox([year_slider_label, year_slider])
        return year_slider, year_slider_box

    
    def _create_plot(self, indicators, year_range):
        df = self._df[self._df['Year'].between(*year_range)]
        plt.figure(figsize=(10, 7))
        years = np.arange(year_range[0], year_range[1]+1,1)
        X = []
        for j in indicators:
            xs = df[df['Country'] ==j]['Gini']
            plt.plot(years,xs, label = j)
        plt.ylabel('Gini',size=16)
        plt.xlabel('Year',size=16)
        plt.legend()
        
    def _on_change(self, _):
        self._update_app()
        
    def _update_app(self):
        indicators = []
        for x in range (0,self.num):
            indicators.append(self._dropdowns[x].value) 
        year_range = self._year_slider.value
        
        #update!
        self._plot_container.clear_output(wait=True)
        with self._plot_container:
            self._create_plot(indicators, year_range)
            plt.show()
        
        


# In[16]:


app4 = App4.from_url(DATA_URL)
app4.container


# In[30]:


#need to refresh this method when you change the previous num!

app5 = App5.from_url(DATA_URL,app4._num)
app5.container


# # lets switch to Lorenz Curves

# In[18]:


def Lorenz_Curve(F,G):

    F = np.sort(F)
    G = np.sort(G)

    F_total = sum(F)
    G_total = sum(G)

    cumul_F = np.cumsum(F).tolist()
    cumul_G = np.cumsum(G).tolist()
    
    portions_F=[0]
    for i in cumul_F:
        portions_F.append(np.round(cumul_F[cumul_F.index(i)]/F_total*100,2))
    portions_G=[0]
    for i in cumul_G:
        portions_G.append(cumul_G[cumul_G.index(i)]/G_total*100)
    
    portions_F = np.round(portions_F,2)
    portions_G = np.round(portions_G,2)

    pop_portionF=np.arange(0,100.1, 100/len(F))
    pop_portionG=np.arange(0,100.1, 100/len(G))
    

    figure(figsize=(12, 8), dpi=80)

    import matplotlib.pyplot as plt
    plt.plot(pop_portionF,portions_F, label='Society 1')
    plt.plot(pop_portionG,portions_G, label = 'Society 2')
    plt.plot([0,100],[0,100], label = 'Perfect Equality')
    plt.legend()
    plt.ylabel('Cumulative Income Share (%)')
    plt.xlabel('Cumulative Population Share (%)')
    plt.title('DIY: Lorenz Curves for Society 1 and 2')
    plt.show()


# In[19]:


def makePops(F):
    F = np.sort(F)
    F_total = sum(F)
    cumul_F = np.cumsum(F).tolist()
    
    portions_F=[0]
    for i in cumul_F:
        portions_F.append(np.round(cumul_F[cumul_F.index(i)]/F_total*100,2))
    
    portions_F = np.round(portions_F,2)
    pop_portionF=np.arange(0,100.1, 100/len(F))
    return pop_portionF, portions_F,


# In[20]:


#Lorenz Curve for 1 society
from matplotlib.pyplot import figure
def single_LC(F):

    F = np.sort(F)
    F_total = sum(F)
    cumul_F = np.cumsum(F).tolist()
    
    portions_F=[0]
    for i in cumul_F:
        portions_F.append(np.round(cumul_F[cumul_F.index(i)]/F_total*100,2))
    
    portions_F = np.round(portions_F,2)
    pop_portionF=np.arange(0,100.1, 100/len(F))

    figure(figsize=(12, 8), dpi=80)

    import matplotlib.pyplot as plt
    plt.plot(pop_portionF,portions_F, label='Society 1')
    plt.plot([0,100],[0,100], label = 'Perfect Equality', color='green')
    plt.legend()
    plt.ylabel('Cumulative Income Share (%)')
    plt.xlabel('Cumulative Population Share (%)')
    plt.title('Lorenz Curve')
    plt.show() 


# In[57]:


#(almost) real-time updating, somewhat trivial content-wise
def LCtophalf():
    def tophalf(percent):
        soc = [100-percent,percent]
      #print('Gini:',gini(soc))
        return single_LC(soc)

    print('Percentage of wealth held by the top half of the population:')
    A1 = widgets.IntSlider(min=50, max=100, step=1, value=75)
    widgets.interact(tophalf, percent=A1)


# In[58]:


LCtophalf()


# In[26]:


class multiLC:
#version2
    
    def __init__(self,n):
        self.num = n
        iv = [1,2,3]
        self._dropdowns = self._create_indicator_dropdown(self.num,iv)

        self._plot_container = widgets.Output()
        
        self._app_container = widgets.VBox([widgets.HTML(
                (
                    """
                    <p>Enter income distrubtions below:

                    """     
                )), widgets.VBox(self._dropdowns), self._plot_container], layout=widgets.Layout(align_items='center', flex='3 0 auto'))
        
        self.container = widgets.VBox([
            widgets.HTML(
                (
                    """<h1>Create your own LCs from income distributions</h1>
                    <p><em>Enter in the income distribution for each society, values seperated by a space</em><p>

                    <p>To change the number of societies, change the number above and rerun this block. 
                    The income distributions do not need to be in order, can have different numbers of people and total income.

                    """     
                ), 
                layout=widgets.Layout(margin='0 0 4em 0')
            ),
            widgets.HBox([
                self._app_container
            ])
        ], layout=widgets.Layout(flex='1 1 auto', margin='0 auto 0 auto', max_width='1024px'))
        self._update_app()     
        
    @classmethod
    def create(cls, n):
        return cls(n)
        
    def _create_indicator_dropdown(self, n, initial_val):
        dropdown = []
        for x in range (0,n):
            s = '1 2 3'
            dropdown.append(widgets.Text(value=s))
            dropdown[x].observe(self._on_change, names=['value'])
        return dropdown      
    
    def _create_plot(self,indicators):
        l = []
        for i in indicators:
            li = i.split()
            li2 = []
            for j in li:
                li2.append(int(j))
            l.append(li2)
        n1 = []
        n2 = [] 
        for i in l:
            n1.append(makePops(i)[0])
            n2.append((makePops(i))[1])

        import matplotlib as plt
        from matplotlib.pyplot import figure
        figure(figsize=(10, 7), dpi=80)
        import matplotlib.pyplot as plt
        i=0
        while i < len(n1):
            s = 'Society' + str(i+1)
            plt.plot(n1[i],n2[i], label=s)
            i+=1

        plt.plot([0,100],[0,100], label = 'Perfect Equality', color='black')
        plt.legend()
        plt.ylabel('Cumulative Income Share (%)')
        plt.xlabel('Cumulative Population Share (%)')
        plt.title('Lorenz Curves')        
        
    def _on_change(self, _):
        self._update_app()
        
    def _update_app(self):
        indicators = []
        for x in range (0,self.num):
            indicators.append(self._dropdowns[x].value) 

        self._plot_container.clear_output(wait=True)
        with self._plot_container:
            self._create_plot(indicators)
            plt.show()
        


# In[23]:


count2 = App4.from_url(DATA_URL)
count2.container


# In[27]:


mlc1 = multiLC.create(count2._num)
mlc1.container


# # Gini

# In[54]:


class Gini:
#version2
    
    def __init__(self,n):
        self.num = n
        iv = [1,2,3]
        self._dropdowns = self._create_indicator_dropdown(self.num,iv)

        self._plot_container = widgets.Output()
        
        self._app_container = widgets.VBox([widgets.HTML(
                (
                    """
                    <p>Enter income distrubtions below:

                    """     
                )), widgets.VBox(self._dropdowns), self._plot_container], layout=widgets.Layout(align_items='center', flex='3 0 auto'))
        
        self.container = widgets.VBox([
            widgets.HTML(
                (
                    """<h1>Find Gini from income distributions</h1>
                    <p><em>Enter in the income distribution for each society, values seperated by a space</em><p>

                    <p>To change the number of societies, change the number above and rerun this block. 
                    The income distributions do not need to be in order, can have different numbers of people and total income.

                    """     
                ), 
                layout=widgets.Layout(margin='0 0 4em 0')
            ),
            widgets.HBox([
                self._app_container
            ])
        ], layout=widgets.Layout(flex='1 1 auto', margin='0 auto 0 auto', max_width='1024px'))
        self._update_app()     
        
    @classmethod
    def create(cls, n):
        return cls(n)
        
    def _create_indicator_dropdown(self, n, initial_val):
        dropdown = []
        for x in range (0,n):
            s = '1 2 3'
            dropdown.append(widgets.Text(value=s))
            dropdown[x].observe(self._on_change, names=['value'])
        return dropdown              
        
    def _on_change(self, _):
        self._update_app()
    
    
    def _calc_Gini(self,indicators):
        l = []
        for i in indicators:
            li = i.split()
            li2 = []
            for j in li:
                li2.append(int(j))
            l.append(li2)
        result = []
        for i in l:
            result.append(GiniArea(i))
        return result
        
    def _update_app(self):
        indicators = []
        for x in range (0,self.num):
            indicators.append(self._dropdowns[x].value) 

        self._plot_container.clear_output(wait=True)
        with self._plot_container:
            r = self._calc_Gini(indicators)
            s = ""
            for i in range (0, len(r)):
                s+="Gini Value " + str(i+1) + ": " + str(np.round(r[i],3)) + '\n'
                
            print (s)
            #plt.show()
        


# In[42]:


count3 = App4.from_url(DATA_URL)
count3.container


# In[55]:


gini1 = Gini.create(count2._num)
gini1.container


# In[ ]:





import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json 
import requests 
from streamlit_lottie import st_lottie 
import pickle
import base64
from sklearn.cluster import KMeans

st.set_page_config(page_title="HR Dashboard", page_icon=":bar_chart:", layout="wide")
# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('HR_Dataset.csv')
    data.drop_duplicates(inplace=True)
    kmeans_df = pd.read_csv('Kmeans.csv')
    with open('HR_model.pkl', 'rb') as file:
        model_svm_loaded = pickle.load(file)
    return data ,model_svm_loaded , kmeans_df

df , model,kmeans_df = load_data()

def plot_ind(avg, name):
    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=avg,
        title={"text": f"{name}"},
        # delta={'reference': reference_value}  # Uncomment and set a reference value if needed
    ))

    fig.add_trace(go.Bar(
        y=[325, 324, 405, 400, 424, 404, 417, 432, 419, 394, 410, 426, 413, 419, 404, 408, 401, 377, 368, 361, 356, 359, 375, 397, 394, 418, 437, 450, 430, 442, 424, 443, 420, 418, 423, 423, 426, 440, 437, 436, 447, 460, 478, 472, 450, 456, 436, 418, 429, 412, 429, 442, 464, 447, 434, 457, 474, 480, 499, 497, 480, 502, 512, 492]
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
    return fig
def perform_kmeans(df, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df[['satisfaction_level', 'last_evaluation', 'average_montly_hours']])
    return cluster_labels
def create_cluster_label_map(num_clusters):
# Basic labels for clustering
    base_labels = ['Beginner', 'Intermediate', 'Advanced', 'Exceptional', 'Expert', 'Master']
    
    # If the number of clusters exceeds the number of base labels, extend the list
    while len(base_labels) < num_clusters:
        base_labels.append(f'Level {len(base_labels) + 1}')
    
    # Create and return the mapping dictionary
    return {i: base_labels[i] for i in range(num_clusters)}
# Set the title of the app
st.markdown("""
        <style>
            .main_container {
                background-color: #FFFFFF;
            }

            h1 {
                text-align: center;
                color: #269BBB;
            }
            h2{
                text-align: center;
               color: #126C85;
            }
         
                  }
            h3{
                text-align: center;
               color: #126C85;
            }
            .stButton>button {
                color: #ffffff;
                background-color: #126C85;
                border: none;
                border-radius: 4px;
                padding: 0.75rem 1.5rem;
                margin: 0.75rem 0;
                position: absolute;
                left:45%;

            }
            .stButton>button:hover {
                background-color:  #269BBB;
                text-align: center;
                color: #FFFFFF;
            }
           
            .stTab{
               background-color:  #269BBB;
            }
            .stTabs [data-baseweb="tab-list"] {
		gap: 30px;
    }
    
          
            body {
            background-color: #F2F2F22;}
        </style>
    """, unsafe_allow_html=True)

if 'submitted' not in st.session_state:
    st.session_state['submitted'] = False
    
st.markdown("""
    <style>
    .floating-form {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }
    </style>
    <div class="floating-form">
        <!-- Your HTML form or content here -->
    </div>
    """, unsafe_allow_html=True)

# Define the welcome page form
if not st.session_state['submitted']:
        
    
        # st.write("Welcome to Eng.Majed AutoMobile Shop! Please submit to continue.")

        # st.markdown("----", unsafe_allow_html=True)
        # st.subheader("Welcome to..")
        colll1,colll2,colll3 = st.columns([0.2,0.4,0.2])
        with colll2:        
         st.title('HR Platform')
        coll1,coll2,coll3 = st.columns([0.3,0.4,0.2])
        with coll2:  
             st.subheader("The PERFECT place to evaluate your employees")     
        col1,col2,col3 = st.columns([0.4,0.4,0.2])
        with col2:
            
            url = requests.get("https://lottie.host/46bee5b7-a234-45f6-b37d-1ef7df4b11e3/qLy06Eb0Tp.json") 
            

            url_json = dict() 
            
            if url.status_code == 200: 
                url_json = url.json() 
            else: 
                print("Error in URL") 


         # st.title("Adding Lottie Animation in Streamlit WebApp") 

            st_lottie(url_json, 
            # change the direction of our animation 
            reverse=True, 
            # height and width of animation 
            height=300, 
            width=300, 
            # speed of animation 
            speed=1, 
            # means the animation will run forever like a gif, and not as a still image 
            loop=True, 
            #  quality of elements used in the animation, other values are "low" and "medium" 
            quality='high', 
            # THis is just to uniquely identify the animation 
            key='Car'
            )
            
        # st.markdown("----", unsafe_allow_html=True)
        # submitted = st.button("Let's GO")
        
        st.markdown("----", unsafe_allow_html=True)
        submitted = st.button("Let's GO") 
        if submitted:
            st.session_state['submitted'] = True
            st.experimental_rerun()  # Rerun the app to update the state

# Define your tabs
if st.session_state['submitted']:
    tab1, tab2,tab3 = st.tabs(["Cluster Analysis", "Prediction","Dashboard :sunglasses:"])
    with tab3:

        st.title('HR Data Analysis Dashboard')
        ################################################################
        ### we should make it change the sidbar input with the changing in the tabs 
        # -- ---- - - -- - - - - - - - - - - -
        
        # Show the first few rows of the dataset
        st.header('Data Overview')
        st.write(df.head())

        # Display summary statistics
        st.header('Summary Statistics')
        st.write(df.describe())
        avg_hours_left = df[df['left'] == 1]['average_montly_hours'].mean()
        avg_hours_stayed = df[df['left'] == 0]['average_montly_hours'].mean()
        avg_time_left = df[df['left'] == 1]['time_spend_company'].mean()
        avg_time_stayed = df[df['left'] == 0]['time_spend_company'].mean()
        st.divider()

        col1, col2,col3= st.columns([0.2,0.2,0.4])
        

        col1.plotly_chart(plot_ind(avg_hours_left,'Average Hours (left)'), use_container_width=True)
        col2.plotly_chart(plot_ind(avg_hours_stayed,'Average Hours (Stayed)'), use_container_width=True)
        col2.plotly_chart(plot_ind(avg_time_left,'Average years (left)'), use_container_width=True)
        col1.plotly_chart(plot_ind(avg_time_stayed,'Average years (Stayed)'), use_container_width=True)
        turnover = df['left'].value_counts().reset_index()
        turnover.columns = ['Status', 'Count']

        # Employee Turnover Analysis
        turnover['Status'] = turnover['Status'].apply(lambda x: 'Left' if x == 1 else 'Stayed')
        turnover_fig = px.pie(turnover, names='Status', values='Count', hole=0.3)
        turnover_fig.update_layout(
            title_text='                                                              Employee Turnover chart', # add a title
            annotations=[dict(text='Turnover', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        col3.plotly_chart(turnover_fig) ## the ring chart 
        ################################################################
        # Employee Distribution by Department

        ########################################################################
        # The input should be here to select which employee are leaving or staying 
        # side bar where you will take input 
        st.sidebar.title('Data Visualization')
        st.sidebar.divider()
        st.sidebar.write("Here you can select the type of the group you want to see their analysis in the dashboard")
        st.sidebar.divider()
        stayed = st.sidebar.checkbox('Stayed')
        left = st.sidebar.checkbox('Left')
        
        ########################################################################

        if left and stayed ==False: 
            df1 = df[df['left'] == 1]
            name = 'left'
        elif stayed and left ==False:
            df1 = df[df['left'] == 0]
            name = 'stayed'
        else:
            df1 = df
            name = 'all'
        st.divider()
        ########################################################################
        # where the indicators will be for the change of the side bar 
        st.title('Select your category')
        st.write("")
        
        i , ii , iii , iiii = st.columns([0.2,0.2,0.2,0.2])
        ### satisfaction_level ,last_evaluation , number_project , Work_accident , promotion_last_5years
        satisfaction_level_avg = df1['satisfaction_level'].mean()
        last_evaluation_avg = df1['last_evaluation'].mean()
        number_project_avg = df1['number_project'].mean()
        promotion_last_5years_avg = df1['promotion_last_5years'].mean()
        st.divider()

        i.plotly_chart(plot_ind(satisfaction_level_avg,f'Satisfaction Level ({name})'), use_container_width=True)
        ii.plotly_chart(plot_ind(last_evaluation_avg,f'Last Evaluation ({name})'), use_container_width=True)
        iii.plotly_chart(plot_ind(number_project_avg,f'Number Project ({name})'), use_container_width=True)
        iiii.plotly_chart(plot_ind(promotion_last_5years_avg,f'Promotions ({name})'), use_container_width=True)
        ########################################################################
        j,jjj= st.columns([0.5,0.5])

        ########################################################################
        # Employee Distribution by Salary Level
        salary_count = df1['salary'].value_counts().reset_index()
        salary_count.columns = ['Salary Level', 'Count']
        

        # Create a horizontal bar chart
        salary_fig = px.bar(salary_count, x='Count', y='Salary Level', orientation='h')

        # Update layout if necessary
        
        salary_fig.update_layout(
            title='Employee Distribution by Salary Level',
            xaxis_title='Count',
            yaxis_title='Salary Level'
        )

        # st.header('Employee Distribution by Salary Level')
        
        j.plotly_chart(salary_fig, use_container_width=True)
        ##################
        # st.header('Employee Distribution by Department')
        
        department_count = df1['time_spend_company'].value_counts().reset_index()
        department_count.columns = ['Years in Company', 'Count']

        years_fig = px.bar(department_count, x='Count', y='Years in Company', orientation='h', 
                        labels={'Years in Company': 'Years in Company', 'Count': 'Count'},
                        title='Employee Distribution by Years')

        jjj.plotly_chart(years_fig, use_container_width=True)

        ########################################################################
        # st.header('Employee Distribution by Department')
        
        department_count = df1['Departments '].value_counts().reset_index()
        department_fig = px.bar(department_count, x='count', y='Departments ', labels={'index': 'Department', 'Departments ': 'Count'},title='Employee Distribution by departments')
        o,oo,ooo= st.columns([0.2,0.5,0.5])
        oo.plotly_chart(department_fig) ### check 
        ########################################################################
        # figgg = px.funnel(df1,x=df1['promotion_last_5years'].value_counts().values[:],y=df1['promotion_last_5years'].unique())
        # j.plotly_chart(figgg)



        ########################################################################
        # Assuming df is your DataFrame and 'left' is the column indicating turnover

        ########################################################################
        st.divider()
        k,kk= st.columns([0.5,0.5])
        ########################################################################

        # Satisfaction and Evaluation Analysis
        
        fig = px.histogram(df1, x='satisfaction_level', marginal='box', nbins=30, title='                                                                                  Satisfaction Level Distribution')
        fig2 = px.histogram(df1, x='last_evaluation', marginal='box', nbins=30, title='                                                                           Last Evaluation Score Distribution')
        k.plotly_chart(fig)
        kk.plotly_chart(fig2)

    with tab2:
        ################################################################
        satisfaction_levels = {
                "😟": 0,     # Represents very dissatisfied (0)
                "🙁": 0.25,  # Represents somewhat dissatisfied (0.25)
                "😐": 0.5,   # Represents neutral (0.5)
                "🙂": 0.75,  # Represents somewhat satisfied (0.75)
                "😄": 1      # Represents very satisfied (1)
            }
        statment = {
            'Yes': 1,
            'No': 0

        }
        salary_emo = {
            '🤏🏻':'low',
            '🤲🏻':'medium',
            '🫶🏻':'high'
        }

        with st.form(key='employee_input_form'):
            col1, col2, col3 = st.columns(3)
        
            with col1:
                selected_emoji = st.radio("Rate employee satisfaction:", list(satisfaction_levels.keys()), horizontal=True)
                evaluation =  st.radio("Rate employee:", list(satisfaction_levels.keys()), horizontal=True)
                accident = st.radio("Any accident ? :", list(statment.keys()), horizontal=True)
                promotion = st.radio("Any promotion on the past 5 years ? :", list(statment.keys()), horizontal=True)
                
                
            with col2:
                number_project = st.number_input('Number of Projects', min_value=0, max_value=10, value=2, step=1)
                time_spend_company = st.number_input('Time Spent in Company (years)', min_value=1, max_value=10, value=3, step=1)
                average_montly_hours = st.number_input('Average Monthly Hours', min_value=90, max_value=310, value=130, step=1)
                
                # work_accident = st.selectbox('Work Accident', options=[0, 1], index=1)
            
            with col3:
                # promotion_last_5years = st.selectbox('Promotion in Last 5 Years', options=[0, 1], index=1)
                departments = st.selectbox('Department', options=['sales', 'technical', 'support', 'IT', 'RandD', 'product_mng',
                                                                'marketing', 'accounting', 'hr', 'management'])
                salar = st.radio("How is the Salary ? :", list(salary_emo.keys()), horizontal=True)
            salary = salary_emo[salar]
            satisfaction_value = satisfaction_levels[selected_emoji]
            last_evaluation =satisfaction_levels[evaluation] 
            work_accident = statment[accident]
            promotion_last_5years=statment[promotion]
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            submit_button = st.form_submit_button(label='Submit')

        # If the form is submitted, you can process the input data here
        if submit_button:
            input_dict = {
    'satisfaction_level': [satisfaction_value],
    'last_evaluation': [last_evaluation],
    'number_project': [number_project],
    'average_montly_hours': [average_montly_hours],
    'time_spend_company': [time_spend_company],
    'Work_accident': [work_accident],
    'promotion_last_5years': [promotion_last_5years],
    'Departments ': [departments],
    'salary': [salary]
}

# Convert the dictionary to a DataFrame
            input_df = pd.DataFrame.from_dict(input_dict)
            # You could include code here to make a prediction with the input data
            # For example:
            prediction = model.predict(input_df)
            probabilities = model.predict_proba(input_df)
            st.write('')
            st.write('')
            st.write('')
            m,mm= st.columns([0.5,0.5])

# You can then display the probability of the class of interest to the user
# If you want to display the probability of the positive class (usually labeled '1')
            fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probabilities[0][0],
            title={'text': "Probability"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [0, 1]}}  # Set the range here
            ))
            fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
            m.plotly_chart(fig)
            if  prediction[0] == 0: 
                majed = 'Stayed'
            else: majed ='Left'

            fig = go.Figure(go.Indicator(
                mode = "number+delta",
                value = prediction[0],
                number = {'prefix': ""},
                domain = {'x': [0, 1], 'y': [0, 1]}
               , title=f'prediction: {majed}'))
            fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10, pad=8),
    )
            # fig.update_layout(paper_bgcolor = "lightgray")
            
            mm.plotly_chart(fig)
            # mm.write(f'{majed}')
    
            st.write('Probability of the classes:', probabilities)
            # And then you could display the prediction
            # st.write(f"The prediction is: {prediction}")
            
            # For now, let's just display the input data as confirmation
            st.dataframe(input_df)

    with tab1:
#         k = st.slider(':blue[ Select the number of clusters (k)]', min_value=2, max_value=6) ### Yes for sure :) 
#         submitted = st.button("Let's GO")
        
#         if submitted:
#             # Prepare the data for clustering
#             dfK = kmeans_df[['satisfaction_level', 'last_evaluation', 'average_montly_hours']]
            

#             # Perform KMeans clustering
#             kmeans = KMeans(n_clusters=k, random_state=42)
#             dfK['cluster_labels'] = kmeans.fit_predict(dfK[['satisfaction_level', 'last_evaluation', 'average_montly_hours']])
#             dfK['cluster_level'] = dfK['cluster_labels'].map({
#     0: 'Beginner',
#     1: 'Intermediate',
#     2: 'Advanced',
#     3: 'Exceptional'
# })
#             # 3D scatter plot using Plotly
#             fig = px.scatter_3d(dfK, 
#                                 x='satisfaction_level', 
#                                 y='last_evaluation', 
#                                 z='average_montly_hours', 
#                                 color='cluster_level', 
#                                 title='3D Cluster Visualization',
#                                 labels={'cluster_labels': 'Cluster'})
            

#             # Display the 3D cluster visualization
#             st.write('')
#             st.write('')
#             st.write('')
#             st.write('')
#             st.plotly_chart(fig)

#         # Get the numeric value of the satisfaction level
        

    # Streamlit UI
        co1,co2,co3= st.columns([0.4,0.5,0.2])
        with co2:
         st.subheader("Select the number of clusters (k)")
        k = st.slider('', min_value=2, max_value=6)
        submitted = st.button("Let's GO")
        if submitted:
            try:
                # Prepare the data for clustering
                dfK = kmeans_df[['satisfaction_level', 'last_evaluation', 'average_montly_hours']]
                
                # Perform KMeans clustering
                dfK['cluster_labels'] = perform_kmeans(dfK, k)

                # Map cluster labels to levels
                cluster_level_map =create_cluster_label_map(k)
                dfK['cluster_level'] = dfK['cluster_labels'].map(cluster_level_map)

                # 3D scatter plot using Plotly
                fig = px.scatter_3d(dfK, x='satisfaction_level', y='last_evaluation', z='average_montly_hours', 
                                    color='cluster_level', title='3D Cluster Visualization')

                # Display the 3D cluster visualization
                st.write('')
                st.write('')
                st.write('')
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                

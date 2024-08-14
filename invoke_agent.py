from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_aws import ChatBedrock
from botocore.config import Config
import boto3
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Retrieve credentials from environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

class SqlAgent:

    def __init__(self, model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0") -> None:
        # Initialize the chat history
        self.chat_history = []
        self.model_id = model_id
        # Initialize the llm
        self.llm = self.get_llm()
        # Initialize db connection
        self.db = self.db_connection()
        # Initialize agent
        self.agent = self.get_agent()

    def db_connection(self):
        region = 'us-east-1'
        athena_url = f"athena.{region}.amazonaws.com" 
        athena_port = '443' #Update, if port is different
        athena_db = 'athena_db' #from user defined params
        s3stagingathena = 's3://athena-destination-store-mped/ ' 
        athena_wkgrp = 'primary' 
        athena_connection_string = f"awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}@{athena_url}:{athena_port}/{athena_db}?s3_staging_dir={s3stagingathena}/&work_group={athena_wkgrp}"
        athena_engine = create_engine(athena_connection_string, echo=True, )
        db = SQLDatabase(athena_engine)

        return db
    
    def get_llm(self):
        # Initialize the language model
        retry_config = Config(
            region_name = 'us-east-1',
            retries = {
                'max_attempts': 100,
                'mode': 'standard'
            }
        )

        boto3_bedrock_runtime = boto3.client("bedrock-runtime", config=retry_config, aws_secret_access_key=aws_secret_access_key, aws_access_key_id=aws_access_key_id)


        model_kwargs =  { 
            "max_tokens": 200000, 
            "temperature": 0,
            "top_k": 250,
            "top_p": 1,
        }
        llm = ChatBedrock(
        client=boto3_bedrock_runtime,
        model_id=self.model_id,
        model_kwargs=model_kwargs,
        )

        return llm
    
    def get_agent(self):

        template = ("""Role: You are a SQL developer creating queries for Amazon Athena to answer non-technical users(Don't use words like sql, database, query and so on) questions about National Accounts Data of Egypt.
                                             
            Objective: Generate SQL queries to return data based on the provided schema and user request.
                                                
            1. Query Decomposition and Understanding:
            - Analyze the user’s request to understand the main objective.
            - Break down reqeusts into sub-queries that can each address a part of the user's request, using the schema provided.
            - If the question does not seem related to the database, just return "I don't know" as the answer.
                        
            2. SQL Query Creation:
            - For each sub-query, use the relevant tables and fields from the provided schema.
            - Construct SQL queries that are precise and tailored to retrieve the exact data required by the user’s request.
            - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
            - Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
            - You can order the results by a relevant column to return the most interesting examples in the database.
                                
            3. Query Execution and Response:
            - Execute the constructed SQL queries against the Amazon Athena database.
            - You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
            - Return the results exactly as they are fetched from the database, ensuring data integrity and accuracy.
            

            Here is the tool you can use: <tool>{tools}</tool>
                    
                   
            Here are columns descriptions for the Amazon Athena database:
            {table_description}


                    
            Relevant pieces of previous conversation:
            <chat_history>{chat_history}</chat_history>
            (You do not need to use these pieces of information if not relevant)
                    
                
            Question: "Question here"
            Thought: You should always think about what to do
            Action: The action to take, should be one of [{tool_names}]
            Action Input: The input to the action
            Observation: The result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: The final answer to the original input question should contain only the information present in the database, without any additional details. It should avoid using technical terms like database, SQL, table, or query. Numbers from the database should not be rounded. The answer should be organized and easy to read, formatted in HTML, and should not use headers greater than h4.
                        
            Question: {input}
                    
            YOU MUST FOLLOW THESE RULES:
                <rules>
                <rule>Sometimes the data for the last year's quarters isn't complete, so when asked about the last quarter, take into consideration checking Q1, Q2, and Q3 instead of just Q4.</rule>
                <rule>If you retrieve data from the private or public sector, or from current or constant prices, you should explicitly state this.</rule>
                <rule>Don't access any data from (governorates_totals_gdp or governorates_activities_gdp) tables, only when a specific governorate is asked by user.</rule>
                <rule>Don't access any data from TotalGrossDomesticProductAtMarketPrices table, only when user asks about market prices.</rule>    
                <rule>When you ask about total gdp general, you must retrieve it from total_value_added table. Give total_value_added the highest periority</rule>
                </rules>

            {agent_scratchpad}   
        """)
        
            # Here are some examples:
            #         <example>- question: What is the growth rate of manufacturing industry in the last quarter?
            #                  - query: SELECT Years, Q1, Q2, Q3, Q4 FROM sectors_growth_rates WHERE Activities = 'ManufacturingIndustries' ORDER BY Years DESC LIMIT 1;
            #         </example>

        prompt = PromptTemplate(
            input_variables=["tools", "top_k", "table_description", "chat_history", "tool_names", "input", "agent_scratchpad"],
            template=template)
        
        agent_executor = create_sql_agent(
            self.llm, 
            db=self.db, 
            verbose=True,
            agent_type="zero-shot-react-description",
            prompt=prompt,
            agent_executor_kwargs={
                "return_intermediate_steps": True,
                "handle_parsing_errors": True,
            },
        )

        return agent_executor
    
    def add_to_chat_history(self, question, answer):
        # Append the new interaction to the history
        self.chat_history.append(f"Human_message: {question}\nAI_message: {answer}\n")
        
        # Keep only the last two interactions
        if len(self.chat_history) > 2:
            self.chat_history.pop(0)

    def clear_chat_history(self):
        self.chat_history = []
            
    def invoke_agent(self, question):
        table_description = """
            <athena_columns>
                <governorates_activities_gdp_columns>
                Governorates: Contains the governorates of Egypt, aggregated governorates in 'Total Egypt' and totals of regions. Here are values of this column: 
                <Governorates>
                    Alexandria,Aswan,Asyut,Beheira,Beni Suef,Cairo,Dakahlia,Damietta,Fayoum,Gharbia,Giza,Ismailia,Kafr ElSheikh,Luxor,Matruh,Minya,Menoufia,New Valley,North Sinai,Port Said,Qalyubia,Qena,Red Sea,Sharqia,Sohag,South Sinai,Suez,Total Egypt,Total Greater Cairo region,Total Central Upper Egypt region,Total Suez Canal region,Total Alexandria region,Total North Upper Egypt region,Total South Upper Egypt region,Total Delta region
                </Governorates>
                Regions: Contains the regions of Egypt, where each region includes many governorates. Here are values of this column:  <Regions>Delta,South Upper Egypt,North Upper Egypt,Alexandria,Suez Canal,Central Upper Egypt,Greater Cairo, Total Egypt</Regions>
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                Activities: Contains the activities or sectors that are in the governorate <Activities>Agriculture,BusinessServices,AccommodationandFoodServiceActivities,Communication,Construction,CrudePetroleumExtraction,DomesticWorkers,Education,ElectricityandGas,GeneralGovernment,FinancialCorporations,Health,Information,ManufacturingIndustries,NonFinancialCorporations,NonProfitInstitutionsServingHouseholdSector,OtherExtractions,OtherServices,PetroleumRefinement,RealEstateOwnership,Sewerage,WasteRecycling,Water,WholesaleandRetailTrade</Activities>
                GDP_Per_Activity: Contains the gross domestic product GDP values for each activity, and it is mandatory to measure any value in units of 'Thousand EGP' when retrieving data from this column.
                </governorates_activities_gdp_columns>
                <governorates_totals_gdp_columns>
                Governorates: Contains the governorates of Egypt, aggregated governorates in 'Total Egypt' and totals of regions. Here are values of this column: 
                <Governorates>
                    Alexandria,Aswan,Asyut,Beheira,Beni Suef,Cairo,Dakahlia,Damietta,Fayoum,Gharbia,Giza,Ismailia,Kafr ElSheikh,Luxor,Matruh,Minya,Menoufia,New Valley,North Sinai,Port Said,Qalyubia,Qena,Red Sea,Sharqia,Sohag,South Sinai,Suez,Total Egypt,Total Greater Cairo region,Total Central Upper Egypt region,Total Suez Canal region,Total Alexandria region,Total North Upper Egypt region,Total South Upper Egypt region,Total Delta region
                </Governorates>
                Regions: Contains the regions of Egypt, where each region includes many governorates. Here are values of this column:  <Regions>Delta,South Upper Egypt,North Upper Egypt,Alexandria,Suez Canal,Central Upper Egypt,Greater Cairo, Total Egypt</Regions>
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                Total_GDP: Contains gross domestic product (GDP) values for aggregated activities in a specific year. And it is the main column to get the total GDP. And it is mandatory to measure any value in units of 'Thousand EGP' when retrieving data from this column.
                Custom_Fees: Contains gross domestic product (GDP) values for custom fees in a specific year. And it is mandatory to measure any value in units of 'Thousand EGP' when retrieving data from this column.
                Total_GDP_With_Custom_Fees: Contains gross domestic product (GDP) values for custom fees plus total GDP in a specific year. And it is mandatory to measure any value in units of 'Thousand EGP' when retrieving data from this column.
                </governorates_totals_gdp_columns>
                <investments_activities_columns>
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                Activities: Contains economic activities or sectors for public investments<Activities>AccommodationandFoodServiceActivities,Agriculture,Construction,Education,Electricity,FinancialIntermediaryInsuranceAndSocialSecurity,Health,InformationAndCommunication,NaturalGas,OtherExtractions,OtherManufacturing,OtherServices,Petroleum,RealEstateOwnership,SuezCanal,TransportationAndStorage,WaterAndSewerage,WholesaleandRetailTrade,petroleumRefining</Activities>
                First_Quarter: Contains the first-quarter public investments at current prices for a specific activity in a specific year. And it is mandatory to measure any value in units of 'Million EGP' when retrieving data from this column.
                Second_Quarter: Contains the second-quarter public investments at current prices for a specific activity in a specific year. And it is mandatory to measure any value in units of 'Million EGP' when retrieving data from this column.
                Third_Quarter: Contains the third-quarter public investments at current prices for a specific activity in a specific year. And it is mandatory to measure any value in units of 'Million EGP' when retrieving data from this column.
                Fourth_Quarter: Contains the fourth-quarter public investments at current prices for a specific activity in a specific year. And it is mandatory to measure any value in units of 'Million EGP' when retrieving data from this column.
                Total_Activity_Investment: Contains the total public investments at current prices for a specific activity in a specific year. And it is mandatory to measure any value in units of 'Million EGP' when retrieving data from this column.
                </investments_activities_columns>
                <investments_totals_columns>
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                First_Quarter: Contains the first-quarter public investments at current prices in a specific year. And it is mandatory to measure any value in units of 'Million EGP' when retrieving data from this column.
                Second_Quarter: Contains the second-quarter public investments at current prices in a specific year. And it is mandatory to measure any value in units of 'Million EGP' when retrieving data from this column.
                Third_Quarter: Contains the third-quarter public investments at current prices in a specific year. And it is mandatory to measure any value in units of 'Million EGP' when retrieving data from this column.
                Fourth_Quarter: Contains the fourth-quarter public investments at current prices in a specific year. And it is mandatory to measure any value in units of 'Million EGP' when retrieving data from this column.
                Total_Year_Investment: Contains the total public investments at current prices in a specific year. And it is mandatory to measure any value in units of 'Million EGP' when retrieving data from this column.
                </investments_totals_columns>
                <expenditure_components_gdp_columns>
                It is mandatory to specify whether the data was retrieved from current price columns or constant price columns.
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                Components: Contains the expenditure components at both market prices and constant prices <components>
                    ExportsOfGoodsAndServices,GovernmentConsumption,GrossCapitalFormation,ImportsOfGoodsAndServices,PrivateConsumption
                </components>
                Q1_Current_Prices: Contains the GDP value at current prices for the first quarter of a specific year and expenditure component. And it is mandatory to measure any value in units of 'Billion EGP' at current prices when retrieving data from this column.
                Q2_Current_Prices: Contains the GDP value at current prices for the second quarter of a specific year and expenditure component. And it is mandatory to measure any value in units of 'Billion EGP' at current prices when retrieving data from this column.
                Q3_Current_Prices: Contains the GDP value at current prices for the third quarter of a specific year and expenditure component. And it is mandatory to measure any value in units of 'Billion EGP' at current prices when retrieving data from this column.
                Q4_Current_Prices: Contains the GDP value at current prices for the fourth quarter of a specific year and expenditure component. And it is mandatory to measure any value in units of 'Billion EGP' at current prices when retrieving data from this column.
                Total_Current_Prices: Contains the GDP value at current prices for a specific year and expenditure component. And it is mandatory to measure any value in units of 'Billion EGP' at current prices when retrieving data from this column.
                Q1_Constant_Prices: Contains the GDP value at constant prices (base year 2021-2022) for the first quarter of a specific year and expenditure component. And it is mandatory to measure any value in units of 'Billion EGP' at constant prices when retrieving data from this column.
                Q2_Constant_Prices: Contains the GDP value at constant prices (base year 2021-2022) for the second quarter of a specific year and expenditure component. And it is mandatory to measure any value in units of 'Billion EGP' at constant prices when retrieving data from this column.
                Q3_Constant_Prices: Contains the GDP value at constant prices (base year 2021-2022) for the third quarter of a specific year and expenditure component. And it is mandatory to measure any value in units of 'Billion EGP' at constant prices when retrieving data from this column.
                Q4_Constant_Prices: Contains the GDP value at constant prices (base year 2021-2022) for the fourth quarter of a specific year and expenditure component. And it is mandatory to measure any value in units of 'Billion EGP' at constant prices when retrieving data from this column.
                Total_Constant_Prices: Contains the GDP value at constant prices (base year 2021-2022) for a specific year and expenditure component. And it is mandatory to measure any value in units of 'Billion EGP' at constant prices when retrieving data from this column.
                </expenditure_components_gdp_columns>
                <TotalGrossDomesticProductAtMarketPrices_columns>
                It is mandatory to specify whether the data was retrieved from current price columns or constant price columns.
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                Q1_Current_Prices: Contains the total gross domestic product at market prices value at current prices for the first quarter of a specific year. And it is mandatory to measure any value in units of 'Billion EGP' at current prices when retrieving data from this column.
                Q2_Current_Prices: Contains the total gross domestic product at market prices value at current prices for the second quarter of a specific year. And it is mandatory to measure any value in units of 'Billion EGP' at current prices when retrieving data from this column.
                Q3_Current_Prices: Contains the total gross domestic product at market prices value at current prices for the third quarter of a specific year. And it is mandatory to measure any value in units of 'Billion EGP' at current prices when retrieving data from this column.
                Q4_Current_Prices: Contains the total gross domestic product at market prices value at current prices for the fourth quarter of a specific year. And it is mandatory to measure any value in units of 'Billion EGP' at current prices when retrieving data from this column.
                Total_Current_Prices: Contains the total gross domestic product at market prices value at current prices for a specific year. And it is mandatory to measure any value in units of 'Billion EGP' at current prices when retrieving data from this column.
                Q1_Constant_Prices: Contains the total gross domestic product at market prices value at constant prices for the first quarter of a specific year. And it is mandatory to measure any value in units of 'Billion EGP' at constant prices when retrieving data from this column.
                Q2_Constant_Prices: Contains the total gross domestic product at market prices value at constant prices for the second quarter of a specific year. And it is mandatory to measure any value in units of 'Billion EGP' at constant prices when retrieving data from this column.
                Q3_Constant_Prices: Contains the total gross domestic product at market prices value at constant prices for the third quarter of a specific year. And it is mandatory to measure any value in units of 'Billion EGP' at constant prices when retrieving data from this column.
                Q4_Constant_Prices: Contains the total gross domestic product at market prices value at constant prices for the fourth quarter of a specific year. And it is mandatory to measure any value in units of 'Billion EGP' at constant prices when retrieving data from this column.
                Total_Constant_Prices: Contains the total gross domestic product at market prices value at constant prices for a specific year. And it is mandatory to measure any value in units of 'Billion EGP' at constant prices when retrieving data from this column.
                </TotalGrossDomesticProductAtMarketPrices_columns>
                <real_gdp_growth_rates_columns>
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                Q1: Contains the real GDP growth rate value at market prices for the first quarter of a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q2: Contains the real GDP growth rate value at market prices for the second quarter of a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q3: Contains the real GDP growth rate value at market prices for the third quarter of a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q4: Contains the real GDP growth rate value at market prices for the fourth quarter of a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Total: Contains the total real GDP growth rate value at market prices of a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                </real_gdp_growth_rates_columns>
                <total_gdp_growth_rate_at_factor_cost_columns>
                It is mandatory to specify whether the data was retrieved from public sector columns or private sector columns.
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                Public_Q1: Contains the total GDP growth rate at factor cost for the first quarter of the public sector for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Public_Q2: Contains the total GDP growth rate at factor cost for the second quarter of the public sector for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Public_Q3: Contains the total GDP growth rate at factor cost for the third quarter of the public sector for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Public_Q4: Contains the total GDP growth rate at factor cost for the fourth quarter of the public sector for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Public: Contains the total GDP growth rate at factor cost for the public sector for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Private_Q1: Contains the total GDP growth rate at factor cost for the first quarter of the private sector for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Private_Q2: Contains the total GDP growth rate at factor cost for the second quarter of the private sector for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Private_Q3: Contains the total GDP growth rate at factor cost for the third quarter of the private sector for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Private_Q4: Contains the total GDP growth rate at factor cost for the fourth quarter of the private sector for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Private: Contains the total GDP growth rate at factor cost for the private sector for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q1: Contains the total GDP growth rate at factor cost for the first quarter of a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q2: Contains the total GDP growth rate at factor cost for the second quarter of a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q3: Contains the total GDP growth rate at factor cost for the third quarter of a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q4: Contains the total GDP growth rate at factor cost for the fourth quarter of a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Total: Contains the total GDP growth rate at factor cost for a specific year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                </total_gdp_growth_rate_at_factor_cost_columns>
                <sectors_growth_rates_columns>
                It is mandatory to specify whether the data was retrieved from public sector columns or private sector columns.
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                Activities: Contains economic activities or sectors of growth rates<Activities>AccommodationandFoodServiceActivities,AgricultureForestryFishing,BusinessServices,Communication,Construction,Education,Electricity,FinancialIntermediariesAuxiliaryServices,Gas,GeneralGovernment,Health,Information,ManufacturingIndustries,MiningQuarrying,OtherExtractions,OtherManufacturing,OtherServices,Petroleum,RealEstateActivitie,RealEstateOwnership,SocialSecurityAndInsurance,SocialServices,SuezCanal,TransportationAndStorage,WaterSewerageRemediationActivitie,WholesaleandRetailTrade,petroleumRefining</Activities>.
                Public_Q1: Contains the GDP growth rate at factor cost for the first quarter of the public sector for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Public_Q2: Contains the GDP growth rate at factor cost for the second quarter of the public sector for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Public_Q3: Contains the GDP growth rate at factor cost for the third quarter of the public sector for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Public_Q4: Contains the GDP growth rate at factor cost for the fourth quarter of the public sector for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Public: Contains the GDP growth rate at factor cost for the public sector for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Private_Q1: Contains the GDP growth rate at factor cost for the first quarter of the private sector for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Private_Q2: Contains the GDP growth rate at factor cost for the second quarter of the private sector for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Private_Q3: Contains the GDP growth rate at factor cost for the third quarter of the private sector for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Private_Q4: Contains the GDP growth rate at factor cost for the fourth quarter of the private sector for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Private: Contains the GDP growth rate at factor cost for the private sector for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q1: Contains the GDP growth rate at factor cost for the first quarter of a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q2: Contains the GDP growth rate at factor cost for the second quarter of a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q3: Contains the GDP growth rate at factor cost for the third quarter of a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Q4: Contains the GDP growth rate at factor cost for the fourth quarter of a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                Total: Contains the GDP growth rate at factor cost for a specific activity and year. And it is mandatory to measure any value in units of '%' at constant prices when retrieving data from this column.
                </sectors_growth_rates_columns>
                <activity_value_added_columns>
                It is mandatory to specify whether the data was retrieved from public sector columns or private sector columns.
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                Activities: Contains economic activities or sectors of gross value added at factor cost<Activities>AccommodationandFoodServiceActivities,AgricultureForestryFishing,BusinessServices,Communication,Construction,Education,Electricity,FinancialIntermediariesAuxiliaryServices,Gas,GeneralGovernment,Health,Information,ManufacturingIndustries,MiningQuarrying,OtherExtractions,OtherManufacturing,OtherServices,Petroleum,RealEstateActivitie,RealEstateOwnership,SocialSecurityAndInsurance,SocialServices,SuezCanal,TransportationAndStorage,WaterSewerageRemediationActivitie,WholesaleandRetailTrade,petroleumRefining</Activities>.
                Public_Q1: Contains the gross value added at factor cost for the first quarter of the public sector for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Public_Q2: Contains the gross value added at factor cost for the second quarter of the public sector for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Public_Q3: Contains the gross value added at factor cost for the third quarter of the public sector for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Public_Q4: Contains the gross value added at factor cost for the fourth quarter of the public sector for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Public: Contains the gross value added at factor cost for the public sector for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Private_Q1: Contains the gross value added at factor cost for the first quarter of the private sector for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Private_Q2: Contains the gross value added at factor cost for the second quarter of the private sector for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Private_Q3: Contains the gross value added at factor cost for the third quarter of the private sector for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Private_Q4: Contains the gross value added at factor cost for the fourth quarter of the private sector for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Private: Contains the gross value added at factor cost for the private sector for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Q1: Contains the gross value added at factor cost for the first quarter of a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Q2: Contains the gross value added at factor cost for the second quarter of a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Q3: Contains the gross value added at factor cost for the third quarter of a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Q4: Contains the gross value added at factor cost for the fourth quarter of a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Total: Contains the total gross value added at factor cost for a specific activity and year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                </activity_value_added_columns>
                <total_value_added_columns>
                It is mandatory to specify whether the data was retrieved from public sector columns or private sector columns.
                Years: Contains financial years that should be treated as string data types, formatted as 'year/(year-1)'. For example: '2024/2023', '2020/2019', and so on. If the user enters a year like '2020' or '2019/2020', it should be considered as '2020/2019'. Similarly '2016' or  '2015/2016' should be considered as '2016/2015', and so on.
                Public_Q1: Contains the gross value added at factor cost for the first quarter of the public sector for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Public_Q2: Contains the gross value added at factor cost for the second quarter of the public sector for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Public_Q3: Contains the gross value added at factor cost for the third quarter of the public sector for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Public_Q4: Contains the gross value added at factor cost for the fourth quarter of the public sector for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Public: Contains the gross value added at factor cost for the public sector for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Private_Q1: Contains the gross value added at factor cost for the first quarter of the private sector for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Private_Q2: Contains the gross value added at factor cost for the second quarter of the private sector for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Private_Q3: Contains the gross value added at factor cost for the third quarter of the private sector for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Private_Q4: Contains the gross value added at factor cost for the fourth quarter of the private sector for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Private: Contains the gross value added at factor cost for the private sector for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Q1: Contains the gross value added at factor cost for the first quarter of a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Q2: Contains the gross value added at factor cost for the second quarter of a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Q3: Contains the gross value added at factor cost for the third quarter of a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Q4: Contains the gross value added at factor cost for the fourth quarter of a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                Total: Contains the total gross value added at factor cost for a specific year. And it is mandatory to measure any value in units of 'Million EGP' at current prices when retrieving data from this column.
                </total_value_added_columns>
            </athena_columns>
        """        
        # Prepare the input data for the agent
        input_data = {
            "input": question,
            "table_description": table_description,
            "top_k": 4,  # Adjust as needed
            # "agent_scratchpad": "",
            "chat_history": "\n".join(self.chat_history),
            "tools": "sql_db_query"
        }

        # Invoke the agent
        response = self.agent.invoke(input_data, verbose=True)
        # Extract the query
        # query = response.get("intermediate_steps")[-1][0].tool_input
        findal_response = response.get("output")
        # Update chat history
        self.add_to_chat_history(question, findal_response)
        
        return response, self.chat_history, response.get("intermediate_steps")
    

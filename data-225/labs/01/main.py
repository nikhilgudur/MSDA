import pandas as pd
import numpy as np

companies = pd.read_csv('./dataset/company_details/companies.csv')
company_industries = pd.read_csv(
    './dataset/company_details/company_industries.csv')
company_specialties = pd.read_csv(
    './dataset/company_details/company_specialities.csv')
employee_counts = pd.read_csv('./dataset/company_details/employee_counts.csv')

benefits = pd.read_csv('./dataset/job_details/benefits.csv')
job_industries = pd.read_csv('./dataset/job_details/job_industries.csv')
job_skills = pd.read_csv('./dataset/job_details/job_skills.csv')

job_postings = pd.read_csv('./dataset/job_postings.csv')


print(job_skills.head())

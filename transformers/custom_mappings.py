from transformers.const import MISSING, CAT


# 1. Marital status 
marital_status_map = {
    frozenset({"Married", "Civil Marriage"}): "Married",
    frozenset({"Single", "Divorced", "Widowed"}): "Single",
    frozenset({MISSING}): MISSING,
}


# 2. Residental status 
residental_status_map = {
    frozenset({"Renting"}): "Renting",
    frozenset({"Owner"}): "Owner",
    frozenset({"With Relatives", "With Friends"}): "Squating",
    frozenset({MISSING}): MISSING,
}

# 3. Education 
education_map = {
    frozenset({'Graduate', 'HND'}): 'Graduate',
    frozenset({'Post Graduate'}): 'Post Graduate',
    frozenset({'Primary','Secondary'}): 'Primary and Secondary',
    frozenset({MISSING}): MISSING,
}

features_custom_mappings_dct = {
    CAT + "data_Request_Input_Customer_EducationStatus": education_map,
    CAT + "data_Request_Input_Customer_MaritalStatus": marital_status_map,
    CAT + "data_Request_Input_Customer_ResidentialStatus": residental_status_map,
}
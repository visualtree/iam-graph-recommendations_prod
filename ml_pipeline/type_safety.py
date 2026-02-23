import pandas as pd
import numpy as np

class TypeSafetyManager:
    """Centralized type safety management for ML pipeline"""
    
    @staticmethod
    def standardize_all_data_types(graph_dfs):
        """
        Standardize ALL data types across the entire ML pipeline
        Fixes all type mismatches identified in the audit
        """
        print("🔧 Applying comprehensive type standardization...")
        
        # USERS DataFrame - Fix ALL mismatched types
        if 'users' in graph_dfs:
            df = graph_dfs['users']
            print(f"   📊 Standardizing users DataFrame...")
            
            # Core ID columns
            df['id'] = df['id'].astype('int64')
            
            # Foreign key columns - Convert float64 to nullable int64
            df['ManagerId'] = df['ManagerId'].where(df['ManagerId'].notna()).astype('Int64')
            df['NOrganisationId'] = df['NOrganisationId'].where(df['NOrganisationId'].notna()).astype('Int64') 
            df['NBusinessRoleId'] = df['NBusinessRoleId'].where(df['NBusinessRoleId'].notna()).astype('Int64')
            df['EndpointSystemId'] = df['EndpointSystemId'].where(df['EndpointSystemId'].notna()).astype('Int64')
            df['EmployeeTypeId'] = df['EmployeeTypeId'].where(df['EmployeeTypeId'].notna()).astype('Int64')
            df['HRMSUserId'] = df['HRMSUserId'].where(df['HRMSUserId'].notna()).astype('Int64')
            df['InvalidLoginAttempt'] = df['InvalidLoginAttempt'].where(df['InvalidLoginAttempt'].notna()).astype('Int64')
            df['teamSize'] = df['teamSize'].where(df['teamSize'].notna()).astype('Int64')
            
            # Duplicate ID column fix
            if 'Id' in df.columns:
                df = df.drop('Id', axis=1)  # Remove duplicate 'Id' column
            
            graph_dfs['users'] = df
            print(f"   ✅ Users: Fixed {df.shape[0]} rows")
        
        # ENTITLEMENTS DataFrame - Most critical fix
        if 'entitlements' in graph_dfs:
            df = graph_dfs['entitlements']
            print(f"   📊 Standardizing entitlements DataFrame...")
            
            # Force entitlement IDs to be consistent strings
            df['id'] = df['id'].astype('string')
            df['composite_id'] = df['composite_id'].astype('string')
            df['EndpointSystemId'] = df['EndpointSystemId'].astype('int64')
            
            graph_dfs['entitlements'] = df
            print(f"   ✅ Entitlements: Fixed {df.shape[0]} rows")
        
        # ENTRECON DataFrame - Critical relationship table
        if 'entrecon' in graph_dfs:
            df = graph_dfs['entrecon']
            print(f"   📊 Standardizing entrecon DataFrame...")
            
            # Ensure consistent types for joins
            df['UserId'] = df['UserId'].astype('int64')
            df['EntitlementId'] = df['EntitlementId'].astype('string')
            
            graph_dfs['entrecon'] = df
            print(f"   ✅ Entrecon: Fixed {df.shape[0]} rows")
        
        # ORGANIZATIONS DataFrame
        if 'orgs' in graph_dfs:
            df = graph_dfs['orgs']
            print(f"   📊 Standardizing orgs DataFrame...")
            
            df['id'] = df['id'].astype('int64')
            df['ParentOrgId'] = df['ParentOrgId'].where(df['ParentOrgId'].notna()).astype('Int64')
            df['ScopeId'] = df['ScopeId'].where(df['ScopeId'].notna()).astype('Int64')
            
            # Remove duplicate Id column
            if 'Id' in df.columns:
                df = df.drop('Id', axis=1)
            
            graph_dfs['orgs'] = df
            print(f"   ✅ Orgs: Fixed {df.shape[0]} rows")
        
        # ENDPOINTS DataFrame  
        if 'endpoints' in graph_dfs:
            df = graph_dfs['endpoints']
            print(f"   📊 Standardizing endpoints DataFrame...")
            
            df['id'] = df['id'].astype('int64')
            df['OwnerUserId'] = df['OwnerUserId'].where(df['OwnerUserId'].notna()).astype('Int64')
            df['ServiceAccountId'] = df['ServiceAccountId'].where(df['ServiceAccountId'].notna()).astype('Int64')
            df['EndpointVarianceId'] = df['EndpointVarianceId'].where(df['EndpointVarianceId'].notna()).astype('Int64')
            df['EndpointSystemTypeId'] = df['EndpointSystemTypeId'].astype('int64')
            
            # Remove duplicate Id column
            if 'Id' in df.columns:
                df = df.drop('Id', axis=1)
            
            graph_dfs['endpoints'] = df
            print(f"   ✅ Endpoints: Fixed {df.shape[0]} rows")
        
        # DESIGNATIONS DataFrame
        if 'designations' in graph_dfs:
            df = graph_dfs['designations']
            print(f"   📊 Standardizing designations DataFrame...")
            
            df['id'] = df['id'].astype('int64')
            
            # Remove duplicate Id column
            if 'Id' in df.columns:
                df = df.drop('Id', axis=1)
            
            graph_dfs['designations'] = df
            print(f"   ✅ Designations: Fixed {df.shape[0]} rows")
        
        print("✅ Type standardization complete!")
        return graph_dfs
    
    @staticmethod
    def safe_merge(left_df, right_df, left_on, right_on=None, how='left', suffixes=('', '_right')):
        """Type-safe merge that handles all the common mismatch patterns"""
        
        if right_on is None:
            right_on = left_on
        
        # Handle list of columns for multi-column joins
        if isinstance(left_on, list):
            for i, (l_col, r_col) in enumerate(zip(left_on, right_on)):
                TypeSafetyManager._align_column_types(left_df, right_df, l_col, r_col)
        else:
            TypeSafetyManager._align_column_types(left_df, right_df, left_on, right_on)
        
        return left_df.merge(right_df, left_on=left_on, right_on=right_on, how=how, suffixes=suffixes)
    
    @staticmethod
    def _align_column_types(left_df, right_df, left_col, right_col):
        """Align types between two columns before merge"""
        
        left_type = left_df[left_col].dtype
        right_type = right_df[right_col].dtype
        
        # If types already match, no conversion needed
        if left_type == right_type:
            return
        
        # Handle entitlement ID columns (should be string)
        if 'entitlement' in left_col.lower() or 'entitlement' in right_col.lower():
            left_df[left_col] = left_df[left_col].astype('string')
            right_df[right_col] = right_df[right_col].astype('string')
            return
        
        # Handle user/manager/org ID columns (should be int64, but handle nulls)
        if any(keyword in left_col.lower() for keyword in ['userid', 'managerid', 'organisationid', 'businessroleid']):
            # Convert both to nullable Int64
            left_df[left_col] = left_df[left_col].where(left_df[left_col].notna()).astype('Int64')
            right_df[right_col] = right_df[right_col].where(right_df[right_col].notna()).astype('Int64')
            return
        
        # Default: try to align to the "safer" type
        if left_type == 'object' or right_type == 'object':
            # If either is object, convert both to object
            left_df[left_col] = left_df[left_col].astype('object')
            right_df[right_col] = right_df[right_col].astype('object')
        elif 'int' in str(left_type) and 'float' in str(right_type):
            # int + float -> convert float to nullable int
            right_df[right_col] = right_df[right_col].where(right_df[right_col].notna()).astype('Int64')
        elif 'float' in str(left_type) and 'int' in str(right_type):
            # float + int -> convert float to nullable int  
            left_df[left_col] = left_df[left_col].where(left_df[left_col].notna()).astype('Int64')
    
    @staticmethod
    def safe_filter(df, column, values):
        """Type-safe filtering that handles type mismatches"""
        
        if isinstance(values, (list, set)):
            values = list(values)
        else:
            values = [values]
        
        # Align types
        col_type = df[column].dtype
        
        # Handle entitlement IDs
        if 'entitlement' in column.lower():
            values = [str(v) for v in values]
            df[column] = df[column].astype('string')
        
        # Handle user IDs and other numeric IDs
        elif any(keyword in column.lower() for keyword in ['userid', 'id']) and col_type != 'object':
            values = [int(v) if pd.notna(v) else v for v in values]
        
        return df[df[column].isin(values)]


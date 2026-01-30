import streamlit as st 


def main():
    st.title('Stomatal density app')
    uploaded_files = st.file_uploader('Upload files for analysis', accept_multiple_files=True, type=["jpg", "jpeg", "png"]) 
    # add temp dir 
    # loop over images 
     # make temp file 
     # sent to api 
     # do post processing 
     # get results 
     # (optional) display restults 
     # pd.tocsv 
     # csv ready for download 


    


    st.download_button('download csv file') # check if how what format this can support, if feeded a csv does it keep formatting? 

    return None
    

if __name__ == "__main__":
    main()

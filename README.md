## Agony Aunt App

- Download database data from volume `modal volume get db_data /chat_history.db .` inpsect with the ipynb notebook
- Deploy app on modal: `modal deploy app_name.py`
- Download the model and create a faiss index: `modal run script_name.py`
- `chat_advan_v2_buffer.py` is the most recent version, incoporates a buffer system for cleaner LLM output streams. However, edge case still occurs where spacing is sometimes ommited. 

## Agony Aunt App
- ![image](https://github.com/user-attachments/assets/afd95426-b114-490d-80bf-b450da458e51)
- Built using Modal Labs, vLLM and FastHTML
- Try it out yourself with my demo [here](https://c123ian--rag-chatbot-serve-fasthtml.modal.run)
- Blog post for more details [here](https://c123ian.github.io/posts/aa_Agony-aunt/agony_aunt_blog_post.html)

  
- Download database data from volume `modal volume get db_data /chat_history.db .` inpsect with the ipynb notebook
- Deploy app on modal: `modal deploy app_name.py`
- Download the model and create a faiss index: `modal run script_name.py`
- `chat_advan_v2_buffer.py` is the most recent version, incoporates a buffer system for cleaner LLM output stream. However, edge case still occur where spacing is sometimes ommited. 

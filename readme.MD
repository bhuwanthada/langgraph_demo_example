### Steps to execute the project.
1. make sure you have python 3.10+ version available.
2. Create the virtual environment. ```python3 -m venv <venv_name>```
3. Post successful creation of virtual environment. Run command: ```pip install -r req.txt```
4. Open a new terminal and run command: ```uvicorn main:app --reload```
5. Open a tab in browser and type: ```localhost:8080/docs```. You can able to see the swagger document over there.
6. Hit the POST call and see the execution.

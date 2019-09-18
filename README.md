### Neural WSD

 _   _                      _        _    _ ___________ 
| \ | |                    | |      | |  | /  ___|  _  \
|  \| | ___ _   _ _ __ __ _| |______| |  | \ `--.| | | |
| . ` |/ _ \ | | | '__/ _` | |______| |/\| |`--. \ | | |
| |\  |  __/ |_| | | | (_| | |      \  /\  /\__/ / |/ / 
\_| \_/\___|\__,_|_|  \__,_|_|       \/  \/\____/|___/  
                                                        
                                                        

## Setup
```
git clone https://github.com/osmanbaskaya/neural-wsd.git neural-wsd
cd neural-wsd
pipenv --python 3.7  # assuming that pipenv is already installed. if not, pip install pipenv
pipenv shell
pipenv sync

# In order to contribute
pre-commit install && pre-commit install -t pre-push

# Installing as a library
pip install -e .
```



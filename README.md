### Neural WSD
```
 __    __                                         __          __       __   ______   _______  
|  \  |  \                                       |  \        |  \  _  |  \ /      \ |       \ 
| $$\ | $$  ______   __    __   ______   ______  | $$        | $$ / \ | $$|  $$$$$$\| $$$$$$$\
| $$$\| $$ /      \ |  \  |  \ /      \ |      \ | $$ ______ | $$/  $\| $$| $$___\$$| $$  | $$
| $$$$\ $$|  $$$$$$\| $$  | $$|  $$$$$$\ \$$$$$$\| $$|      \| $$  $$$\ $$ \$$    \ | $$  | $$
| $$\$$ $$| $$    $$| $$  | $$| $$   \$$/      $$| $$ \$$$$$$| $$ $$\$$\$$ _\$$$$$$\| $$  | $$
| $$ \$$$$| $$$$$$$$| $$__/ $$| $$     |  $$$$$$$| $$        | $$$$  \$$$$|  \__| $$| $$__/ $$
| $$  \$$$ \$$     \ \$$    $$| $$      \$$    $$| $$        | $$$    \$$$ \$$    $$| $$    $$
 \$$   \$$  \$$$$$$$  \$$$$$$  \$$       \$$$$$$$ \$$         \$$      \$$  \$$$$$$  \$$$$$$$ 
                                                                                              
```
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



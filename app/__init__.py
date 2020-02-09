from flask import Flask
app = Flask(__name__)

import logging
from time import strftime
logfile = 'logs/info_{}.log'.format(strftime('%Y-%m-%d'))
logformat = '%(asctime)s %(levelname)s %(message)s'

logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO, format=logformat)

logging.info('Starting Viva Las Foodie')

from app import views

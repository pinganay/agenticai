Challenges: USER_AGENT env variable was not being read
Work Around: in terminal, run USER_AGENT="Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15", then run export USER_AGENT

Challenge: Numpy Version was not compatible with other imported APIs
Solution: Downgrade numpy version by uninstalling numpy, then installing numpy 1.26.4
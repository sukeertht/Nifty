mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"sukeertht@gmail.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
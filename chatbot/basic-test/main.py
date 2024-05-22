def get_name():
    username = input('chatter >> Hello, I am chatter. What is your name?\nyou >> ')
    print(f'chatter >> Nice to meet you {username}')
    return username
    
def get_user_input(username):
    user_input = input(f'{username} >> ')
    return user_input

def bot_response(user_input):
    print(f'chatter >> That is nice')

def chat_loop(username):
    user_input = ''
    while user_input.lower() == 'bye':
        user_input = get_user_input(username)
        bot_response(user_input)
    print(f'chatter >> bye')
        
def main():   
    username = get_name()
    chat_loop(username)

if __name__ == "__main__":
    main()
import scipy.io
import numpy as np
import gc


pairs_train = []
pairs_test = []
pairs_social = []
pairs_tag = []
num_train_per_user = 50
user_id = 0
np.random.seed(123)
urls = {}
users = {}
bookmark_url = {}

with open("bookmarks.dat",'rb') as f:
    lines = f.readlines()
    url_id = 0
    for line in lines[1:]:
        line = line.decode('utf-8')
        arr = line.strip().split('\t')
        url = arr[3]
        if(url not in urls.keys()):
            urls[url] = url_id
            url_id += 1
        bookmark = arr[0]
        bookmark_url[bookmark] = urls[url]
    print("num of url: ",url_id)


lines = open("user_taggedbookmarks-timestamps.dat").readlines()
lines = lines[1:]
cur_user = 0
cur_item = 0
user_item_list = []
user_item = []

user_id = 0
for line in lines:
    arr = line.strip().split()
    user = arr[0]
    item = arr[1]
    if(user != cur_user):
        users[user] = user_id
        user_item_list.append(user_item)
        user_item = []
        cur_user = user
        cur_item = item
        user_item.append(bookmark_url[item])
        user_id += 1
    elif(bookmark_url[item] not in user_item):
        user_item.append(bookmark_url[item])
print("num of user:",user_id)  

user_item_list.append(user_item)

user_id = 0
for arr in user_item_list[1:]:
    n = len(arr)
    idx = np.random.permutation(n)
    # assert(n > num_train_per_user)
    for i in range(min(num_train_per_user, n)):
        pairs_train.append([user_id, arr[idx[i]]])
    if n > num_train_per_user:
        for i in range(num_train_per_user, n):
            pairs_test.append([user_id, arr[idx[i]]])
    user_id += 1
num_users = user_id
pairs_train = np.asarray(pairs_train).astype(np.int)
pairs_test = np.asarray(pairs_test).astype(np.int)
print('pairs_train.shape: ',pairs_train.shape)
print('pairs_test.shape: ',pairs_test.shape)
num_items = np.maximum(np.max(pairs_train[:, 1]), np.max(pairs_test[:, 1]))+1
print("num_users=%d, num_items=%d" % (num_users, num_items))

# social communication
with open("user_contacts.dat") as f:
    lines = f.readlines()
    for line in lines[1:]:
        arr = line.strip().split()
        user = arr[0]
        friend = arr[1]
        if((user not in users.keys()) or (friend not in users.keys())):
            pass
        else:
            pairs_social.append([users[user], users[friend]])

pairs_social = np.array(pairs_social).astype(np.int)      
print('pairs_social.shape: ',pairs_social.shape)


tags = {}
with open('tags.dat','rb') as f:
    lines = f.readlines()
    tag_id = 0
    for line in lines[1:]:
        line = line.decode('utf-8')
        arr = line.strip().split()
        tagid = arr[0]
        tags[tagid] = tag_id
        tag_id += 1
    print('num_tags: ',tag_id)


content = np.zeros((url_id,tag_id))
with open('bookmark_tags.dat') as f:
    lines = f.readlines()
    for line in lines[1:]:
        arr = line.strip().split()
        pairs_tag.append([bookmark_url[arr[0]], tags[arr[1]]])
pairs_tag = np.array(pairs_tag).astype(np.int)
print('pairs_tag.shape: ',pairs_tag.shape)



with open("content.dat", "w") as fid:
    for item_id in range(num_items):
        this_item_tag = pairs_tag[pairs_tag[:, 0]==item_id, 1]
        tags_str = " ".join(str(x) for x in this_item_tag)
        fid.write("%d %s\n" % (len(this_item_tag), tags_str))

with open("train_users_{}.dat".format(num_train_per_user), "w") as fid:
    for user_id in range(num_users):
        this_user_items = pairs_train[pairs_train[:, 0]==user_id, 1]
        items_str = " ".join(str(x) for x in this_user_items)
        fid.write("%d %s\n" % (len(this_user_items), items_str))

with open("train_items_{}.dat".format(num_train_per_user), "w") as fid:
    for item_id in range(num_items):
        this_item_users = pairs_train[pairs_train[:, 1]==item_id, 0]
        users_str = " ".join(str(x) for x in this_item_users)
        fid.write("%d %s\n" % (len(this_item_users), users_str))

with open("test_users_{}.dat".format(num_train_per_user), "w") as fid:
    for user_id in range(num_users):
        this_user_items = pairs_test[pairs_test[:, 0]==user_id, 1]
        items_str = " ".join(str(x) for x in this_user_items)
        fid.write("%d %s\n" % (len(this_user_items), items_str))

with open("test_items_{}.dat".format(num_train_per_user), "w") as fid:
    for item_id in range(num_items):
        this_item_users = pairs_test[pairs_test[:, 1]==item_id, 0]
        users_str = " ".join(str(x) for x in this_item_users)
        fid.write("%d %s\n" % (len(this_item_users), users_str))


with open("social.dat", "w") as fid:
    for user_id in range(num_users):
        this_user_friends = pairs_social[pairs_social[:, 0]==user_id, 1]
        friends_str = " ".join(str(x) for x in this_user_friends)
        fid.write("%d %s\n" % (len(this_user_friends), friends_str))

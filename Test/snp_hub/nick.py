#minimize the size of the set
from sortedcontainers import SortedDict
import sys

number_of_snps = 20000000

# thank you chatGPT
def get_full_memory_size(obj, seen=None):
    """Recursively finds the memory size of objects, including nested objects."""
    if seen is None:
        seen = set()

    size = sys.getsizeof(obj)
    obj_id = id(obj)

    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_full_memory_size(k, seen) for k in obj.keys()])
        size += sum([get_full_memory_size(v, seen) for v in obj.values()])
    elif hasattr(obj, '__dict__'):
        size += get_full_memory_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_full_memory_size(i, seen) for i in obj])

    return size

def with_sdict():
    set1 = SortedDict()

    # store all values similar to the dict
    for i in range(number_of_snps):
        set1.update({f"SNP1-SNP{i}-XOR": 1.0})
        set1.update({f"SNP1-SNP{i}-*": 1.0})
        set1.update({f"SNP1-SNP{i}-OR": 1.0})
        set1.update({f"SNP1-SNP{i}-PAGER": 1.0})

    print('with_sdict::size:', get_full_memory_size(set1) /1024/1024/1024) #correctish size
    del set1

def with_sdict_list():
    mappings = {'XOR': 0, '*': 1, 'OR': 2, 'PAGER': 3}
    set1 = SortedDict()

    for i in range(number_of_snps):
        value = [None] * len(mappings)
        for k,v in mappings.items():
            value[v] = 2.0
        set1[f"SNP1-SNP{i}"] = value

    print('with_sdict_list::size:', get_full_memory_size(set1) /1024/1024/1024) #correctish size

    del set1

def with_dict():
    set1 = {}
    for i in range(number_of_snps):
        set1[f"SNP1-SNP{i}"] = {'XOR': 2.0, '*': 2.0, 'OR': 2.0, 'PAGER': 2.0}
    print('with_dict::size:', get_full_memory_size(set1) /1024/1024/1024) #correctish size

    del set1

def with_dict_list():
    mappings = {'XOR': 0, '*': 1, 'OR': 2, 'PAGER': 3}
    set1 = {}

    for i in range(number_of_snps):
        value = [None] * len(mappings)
        for k,v in mappings.items():
            value[v] = 2.0
        set1[f"SNP1-SNP{i}"] = value

    print('with_dict_list::size:', get_full_memory_size(set1) /1024/1024/1024) #correctish size


def main():
    print('with_dict')
    with_dict() # with_dict::size: 4.93696038518101 GB

    print('with_dict_list')
    with_dict_list() # with_dict_list::size: 3.1488208789378405 GB

    print('with_sdict')
    with_sdict() # with_sdict::size: 6.318169057369232 GB

    print('with_sdict_list')
    with_sdict_list() # with_sdict_list::size: 3.1488208938390017 GB


if __name__ == "__main__":
    main()
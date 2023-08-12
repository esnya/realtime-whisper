"""
async_itertools module

This module provides a set of utility functions for working with async iterables.
It includes the following functions:

- afilter(func, iterable): Filters the given async iterable with the provided function.
- amap(func, iterable): Maps the given function onto each item of the async iterable.
- areduce(func, iterable): Reduces the async iterable to a single output with the provided function.
- achain(*iterables): Chains the given async iterables together into a single async iterable.
- amerge(*iterables, stop_on_first_complete=False):
    Merges the given async iterables together into a single async iterable.


This code is the result of a collaboration with OpenAI's ChatGPT
    and is released under the CC0 1.0 Universal (CC0 1.0) Public Domain Dedication.

You can copy, modify, distribute and perform the work, even for commercial purposes, all without asking permission.

For more information, please see <https://creativecommons.org/publicdomain/zero/1.0/>
"""

import asyncio
from typing import Any, AsyncIterable, Callable, Optional, TypeVar, overload


T = TypeVar("T")
U = TypeVar("U")


class EndOfGenerator:
    pass


async def aonce(item: T) -> AsyncIterable[T]:
    """
    Yields a single item.

    :param item: The item to yield.
    :type item: T

    :return: An async iterable that yields the item.
    :rtype: AsyncIterable[T]
    """
    yield item


async def afilter(
    func: Callable[[T], bool], iterable: AsyncIterable[T]
) -> AsyncIterable[T]:
    """
    Filters the given async iterable with the provided function.

    :param func: A function that takes an item and returns a boolean.
    :type func: Callable[[T], bool]
    :param iterable: An async iterable to filter.
    :type iterable: AsyncIterable[T]

    :return: An async iterable that yields the filtered items.
    :rtype: AsyncIterable[T]
    """
    async for i in iterable:
        if func(i):
            yield i


async def amap(func: Callable[[T], U], iterable: AsyncIterable[T]) -> AsyncIterable[U]:
    """
    Maps the given function onto each item of the async iterable.

    :param func: A function that takes an item and returns a new item.
    :type func: Callable[[T], U]
    :param iterable: An async iterable to map over.
    :type iterable: AsyncIterable[T]

    :return: An async iterable that yields the mapped items.
    :rtype: AsyncIterable[U]
    """
    async for i in iterable:
        yield func(i)


@overload
async def areduce(
    func: Callable[[T, T], T],
    iterable: AsyncIterable[T],
) -> T:
    """
    Reduces the async iterable to a single output with the provided function.

    :param func: A function that takes two items and returns a new item.
    :type func: Callable[[T, T], T]
    :param iterable: An async iterable to reduce.
    :type iterable: AsyncIterable[T]

    :return: The reduced item.
    :rtype: T
    """
    pass


@overload
async def areduce(
    func: Callable[[U, T], U],
    iterable: AsyncIterable[T],
    initial: U,
) -> U:
    """
    Reduces the async iterable to a single output with the provided function.

    :param func: A function that takes two items and returns a new item.
    :param func: Callable[[U, T], U]
    :param iterable: An async iterable to reduce.
    :type iterable: AsyncIterable[T]
    :param initial: An initial value for the reduction.
    :type initial: U
    """
    pass


async def areduce(
    func: Callable[[Any, T], Any],
    iterable: AsyncIterable[T],
    initial: Optional[Any] = None,
) -> Any:
    it = iterable.__aiter__()
    if initial is None:
        try:
            result = await it.__anext__()
        except StopAsyncIteration:
            raise TypeError("reduce() of empty sequence with no initial value")
    else:
        result = initial
    async for i in it:
        result = func(result, i)
    return result


# async def areduce(func: Callable[[T, T], T], iterable: AsyncIterable[T]) -> T:
#     """
#     Reduces the async iterable to a single output with the provided function.

#     :param func: A function that takes two items and returns a new item.
#     :type func: Callable[[T, T], T]
#     :param iterable: An async iterable to reduce.
#     :type iterable: AsyncIterable[T]

#     :return: The reduced item.
#     :rtype: T
#     """
#     it = iterable.__aiter__()
#     try:
#         result = await it.__anext__()
#     except StopAsyncIteration:
#         raise TypeError("reduce() of empty sequence with no initial value")
#     async for i in it:
#         result = func(result, i)
#     return result


async def achain(*iterables: AsyncIterable[T]) -> AsyncIterable[T]:
    """
    Chains the given async iterables together into a single async iterable.

    :param iterables: A list of async iterables to chain together.
    :type iterables: AsyncIterable[T]

    :return: An async iterable that yields the chained items.
    :rtype: AsyncIterable[T]
    """
    for it in iterables:
        async for i in it:
            yield i


async def amerge(
    *iterables: AsyncIterable[T], stop_on_first_complete=False
) -> AsyncIterable[T]:
    """
    Merges the given async iterables together into a single async iterable.

    :param iterables: A list of async iterables to merge together.
    :type iterables: AsyncIterable[T]
    :param stop_on_first_complete: If True, stops when the first iterable is exhausted.
    :type stop_on_first_complete: bool (default: False)

    :return: An async iterable that yields the merged items.
    :rtype: AsyncIterable[T]
    """
    queue = asyncio.Queue()

    async def _gen(it):
        async for i in it:
            await queue.put(i)
        await queue.put(EndOfGenerator())

    tasks = [asyncio.create_task(_gen(it)) for it in iterables]

    while tasks:
        result = await queue.get()
        if isinstance(result, EndOfGenerator):
            if stop_on_first_complete:
                for task in tasks:
                    task.cancel()
                break
            else:
                tasks = [task for task in tasks if not task.done()]
        else:
            yield result

"""Bounded, run-scoped caches; failed operations never poison later retries."""
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from hashlib import sha256
from threading import Lock
import time
from numbers import Real
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def canonical_url(url):
    """Remove fragments and tracking parameters, preserving semantic queries."""
    parts = urlsplit(url)
    if parts.scheme not in ('http', 'https'):
        return url
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith('utm_') and k.lower() not in ('fbclid', 'gclid')]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or '/',
                       urlencode(query), ''))


def content_hash(text):
    return sha256((text or '').encode('utf-8')).hexdigest()


class SingleFlightCache:
    def __init__(self, max_entries=128):
        self.max_entries = max_entries
        self._values = OrderedDict()
        self._pending = {}
        self._lock = Lock()
        self.hits = self.misses = 0

    def clear(self):
        with self._lock:
            self._values.clear()
            self.hits = self.misses = 0

    def get_or_compute(self, key, compute, cacheable=lambda value: True, wait_timeout=None):
        with self._lock:
            if key in self._values:
                self.hits += 1
                self._values.move_to_end(key)
                return deepcopy(self._values[key])
            future = self._pending.get(key)
            owner = future is None
            if owner:
                self.misses += 1
                future = self._pending[key] = Future()
            else:
                self.hits += 1
        if not owner:
            return deepcopy(future.result(timeout=wait_timeout))
        try:
            result = compute()
            with self._lock:
                if cacheable(result):
                    self._values[key] = deepcopy(result)
                    while len(self._values) > self.max_entries:
                        self._values.popitem(last=False)
            future.set_result(result)
            return deepcopy(result)
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._pending.pop(key, None)


class CachedSearchClient:
    """Delegate search/settings while sharing immutable fetched pages per run."""
    def __init__(self, client, max_entries=128):
        object.__setattr__(self, '_client', client)
        object.__setattr__(self, '_page_cache', SingleFlightCache(max_entries))
        object.__setattr__(self, '_timings', None)

    def __getattr__(self, name):
        return getattr(self._client, name)

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            setattr(self._client, name, value)

    def clear_cache(self):
        self._page_cache.clear()

    def _call(self, stage, operation, *args, **kwargs):
        if self._timings is None:
            return operation(*args, **kwargs)
        return self._timings.call(stage, operation, *args, **kwargs)

    def search(self, *args, **kwargs):
        return self._call('search', self._client.search, *args, **kwargs)

    def get_page_content(self, url, **kwargs):
        refresh = kwargs.pop('refresh', False)
        if refresh:
            return self._call('fetch', self._client.get_page_content, url, **kwargs)
        timeout = kwargs.get('timeout', getattr(self._client, 'timeout', None))
        wait_timeout = float(timeout) if isinstance(timeout, Real) else None
        if isinstance(kwargs.get('deadline'), Real):
            remaining = kwargs['deadline'] - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('page fetch deadline exhausted')
            wait_timeout = min(wait_timeout, remaining) if wait_timeout is not None else remaining
            kwargs['timeout'] = wait_timeout
        if wait_timeout is not None and wait_timeout <= 0:
            raise ValueError('fetch timeout must be positive')
        key = (canonical_url(url), tuple(sorted((k, repr(v)) for k, v in kwargs.items()
                                               if k not in ('timeout', 'deadline'))))
        page = self._page_cache.get_or_compute(
            key, lambda: self._call('fetch', self._client.get_page_content, url, **kwargs),
            cacheable=lambda value: (bool(getattr(value, 'text_content', ''))
                                     and not (getattr(value, 'metadata', {}) or {}).get('error')),
            wait_timeout=wait_timeout,
        )
        return page

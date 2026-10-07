#!/usr/bin/env python3
"""VRCTimetable Discord robot (Stage 11C, 2026-10-05).

Reads the Scheduled Events of every Discord server that has the organiser's bot, and writes them to
discord.json for the VRCTimetable sheet, which turns them into board events (the sheet decides which
servers are shown, and how). Google's script service cannot call Discord's bot API (Discord refuses
it), so this runs in GitHub Actions in the organiser's own repository (workflow
vrctimetable-discord.yml, started by the sheet). Python standard library only.

    DISCORD_BOT_TOKEN=... DISCORD_DISPATCHED=true DISCORD_SERVERS=<id>,<id> DISCORD_ASK=<id>,…
    DISCORD_HIDDEN=<event id>,… DISCORD_APPROVED=<event id>:<version>,… python fetch_discord.py out [previous]

The bot key comes ONLY from the environment (a repository secret) and is never printed or written.
discord.json lands in a PUBLIC repository, and Discord events are meant for a server's members, so
(all lists come from the sheet when it starts the robot):
  - events are read only for the servers the organiser ticked "Show" (DISCORD_SERVERS); of any other
    server only its id and name are written (the sheet lists it as waiting for the tick);
  - an event the organiser hid (DISCORD_HIDDEN), or of an "Ask me first" server (DISCORD_ASK) whose
    version the organiser has not said OK to (DISCORD_APPROVED), goes out with name and dates only.
Reliability: a server Discord does not answer for keeps its events from the last result; a failed
look at the server list keeps the whole last result; a run not started by the sheet (no lists: the
Run button, a change of this file) keeps the last result when there is one. Only what the board
needs is kept: no member counts, no creator. The file changes only when the events change (sorted, no
time stamp), so the workflow commits nothing on a quiet run.

discord.json:
    {"v": 1, "status": "ok" | "nokey" | "badkey" | "error|<details>",
     "bot": {"id": "<application id>", "name": "<bot name>"},
     "servers": [{"id", "name", "icon", "read": true | false, "status": "ok" | "noaccess" | "error|<details>",
                  "events": [{"id", "name", "description", "start", "end", "status", "type",
                              "location", "channel", "image", "rule", "exceptions": [...]}
                             or {"id", "name", "start", "end", "status", "type", "rule", "exceptions",
                                 "version", "hidden" | "pending": true}]}]}
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

API = 'https://discord.com/api/v10'
MAX_SERVERS = 100       # an unverified bot cannot join more
MAX_RETRIES = 3
MAX_WAIT = 30.0         # seconds: Discord asks to wait (429) — longer waits are given up on this run


def user_agent():
    repo = os.environ.get('GITHUB_REPOSITORY', 'vrctimetable-pictures')
    # Discord's rule: "DiscordBot ($url, $versionNumber)"; other agents are refused.
    return 'DiscordBot (https://github.com/' + repo + ', 1.0)'


def http_get(url, token, timeout=20):
    """(code, parsed JSON or None). Network trouble = (0, None)."""
    req = urllib.request.Request(url, headers={
        'Authorization': 'Bot ' + token,
        'User-Agent': user_agent(),
        'Accept': 'application/json',
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode('utf-8'))
        except Exception:
            body = None
        return e.code, body
    except Exception:
        return 0, None


class Discord:
    """Calls with Discord's rate limits respected (429: wait retry_after, a few times)."""

    def __init__(self, token, get=http_get, sleep=time.sleep):
        self.token = token
        self.get = get
        self.sleep = sleep

    def call(self, path):
        for attempt in range(MAX_RETRIES + 1):
            code, body = self.get(API + path, self.token)
            if code == 429 and attempt < MAX_RETRIES:
                wait = 1.0
                if isinstance(body, dict) and isinstance(body.get('retry_after'), (int, float)):
                    wait = float(body['retry_after'])
                if wait > MAX_WAIT:
                    return code, body
                self.sleep(wait + 0.25)
                continue
            if code == 0 and attempt < 1:
                self.sleep(2.0)
                continue
            return code, body
        return code, body


def text(v, limit):
    return v[:limit] if isinstance(v, str) else ''


def snowflake_key(x):
    try:
        return int(x.get('id', 0))
    except (TypeError, ValueError):
        return 0


def clean_rule(rule):
    """The recurrence rule as Discord gives it, without empty fields (stable text)."""
    if not isinstance(rule, dict):
        return None
    keep = ('start', 'end', 'frequency', 'interval', 'by_weekday', 'by_n_weekday', 'by_month', 'by_month_day', 'by_year_day', 'count')
    out = {}
    for k in keep:
        v = rule.get(k)
        if v is not None and v != []:
            out[k] = v
    return out


def clean_exceptions(ev):
    out = []
    for x in ev.get('guild_scheduled_event_exceptions') or []:
        if not isinstance(x, dict) or not x.get('event_exception_id'):
            continue
        item = {'id': str(x['event_exception_id'])}
        if x.get('scheduled_start_time'):
            item['start'] = x['scheduled_start_time']
        if x.get('scheduled_end_time'):
            item['end'] = x['scheduled_end_time']
        if x.get('is_canceled'):
            item['canceled'] = True
        out.append(item)
    out.sort(key=lambda i: int(i['id']) if i['id'].isdigit() else 0)
    return out


def clean_event(ev):
    meta = ev.get('entity_metadata') or {}
    return {
        'id': str(ev.get('id', '')),
        'name': text(ev.get('name'), 100),
        'description': text(ev.get('description'), 1000),
        'start': ev.get('scheduled_start_time') or '',
        'end': ev.get('scheduled_end_time') or '',
        'status': ev.get('status') if isinstance(ev.get('status'), int) else 0,
        'type': ev.get('entity_type') if isinstance(ev.get('entity_type'), int) else 0,
        'location': text(meta.get('location') if isinstance(meta, dict) else '', 100),
        'channel': str(ev.get('channel_id') or ''),
        'image': text(ev.get('image'), 100),
        'rule': clean_rule(ev.get('recurrence_rule')),
        'exceptions': clean_exceptions(ev),
    }


def allowed_servers(text):
    """DISCORD_SERVERS ("<id>,<id>…") → the set of server ids whose events may be read (ids only: digits)."""
    out = set()
    for part in (text or '').split(','):
        part = part.strip()
        if part.isdigit() and 5 <= len(part) <= 25:
            out.add(part)
    return out


def approved_versions(text):
    """DISCORD_APPROVED ("<event id>:<64 hex>,…") → {event id: version the organiser said OK to}."""
    out = {}
    for part in (text or '').split(','):
        bits = part.strip().split(':')
        if len(bits) == 2 and bits[0].isdigit() and 5 <= len(bits[0]) <= 25 and len(bits[1]) == 64 and all(c in '0123456789abcdef' for c in bits[1]):
            out[bits[0]] = bits[1]
    return out


def version_of(ev):
    """
    The hash the sheet keeps for an OK (its discordVersion_): SHA-256 over the SHA-256 of each of name,
    description, image and location, 64 hex. Each field on its own, so text cannot move between fields
    unseen; the whole digest, so nobody can search for a harmless and a hostile text with the same
    version (12 hex digits, until 2026-10-07, could be matched in seconds).
    """
    parts = ''.join(hashlib.sha256(str(ev.get(k) or '').encode('utf-8', 'replace')).hexdigest()
                    for k in ('name', 'description', 'image', 'location'))
    return hashlib.sha256(parts.encode('ascii')).hexdigest()


def only_what_approval_needs(ev, flag):
    """An event that is hidden, or waits for the organiser's OK: name and dates only (the file is public)."""
    keep = {k: ev[k] for k in ('id', 'name', 'start', 'end', 'status', 'type', 'rule', 'exceptions')}
    keep['version'] = version_of(ev)
    keep[flag] = True
    return keep


def fetch(token, discord=None, allowed=frozenset(), ask=frozenset(), hidden=frozenset(), approved=None, previous=None):
    """
    Everything the sheet needs, as one dict (see the module notes). allowed = shown servers (their
    events are read); ask = shown servers with "Ask me first": an event goes out in full only when its
    version matches the organiser's OK (approved), else with name and dates only; hidden = event ids the
    organiser hid (name and dates only). previous = the last result: a server that fails this time keeps
    its events from it.
    """
    approved = approved or {}
    before = {}
    for s in (previous or {}).get('servers') or []:
        if isinstance(s, dict) and s.get('id'):
            before[str(s['id'])] = s
    out = {'v': 1, 'status': 'ok', 'bot': {}, 'servers': []}
    if not token:
        out['status'] = 'nokey'
        return out
    d = discord or Discord(token)
    code, app = d.call('/oauth2/applications/@me')
    if code == 401:
        out['status'] = 'badkey'
        return out
    if code == 200 and isinstance(app, dict):
        out['bot'] = {'id': str(app.get('id', '')), 'name': text(app.get('name'), 100)}
    guilds = []
    after = None
    while len(guilds) < MAX_SERVERS:
        path = '/users/@me/guilds?limit=200' + ('&after=' + after if after else '')
        code, page = d.call(path)
        if code == 401:
            out['status'] = 'badkey'
            return out
        if code != 200 or not isinstance(page, list):
            out['status'] = 'error|servers ' + str(code)
            return out
        guilds.extend(g for g in page if isinstance(g, dict) and g.get('id'))
        if len(page) < 200:
            break
        after = str(page[-1].get('id'))
    for g in sorted(guilds[:MAX_SERVERS], key=snowflake_key):
        server = {'id': str(g['id']), 'name': text(g.get('name'), 100), 'icon': '', 'read': False, 'status': 'ok', 'events': []}
        out['servers'].append(server)
        if server['id'] not in allowed:
            continue    # not shown by the organiser: only its name (the sheet lists it as waiting) — no events, no icon
        server['read'] = True
        server['icon'] = text(g.get('icon'), 100)
        code, events = d.call('/guilds/' + server['id'] + '/scheduled-events')
        if code == 401:
            out['status'] = 'badkey'
            return out
        if code in (403, 404):
            server['status'] = 'noaccess'
        elif code != 200 or not isinstance(events, list):
            # Discord did not answer for this server: its events from the last look stay (no gap on the boards).
            server['status'] = 'error|' + str(code)
            old = before.get(server['id'])
            if old and isinstance(old.get('events'), list):
                server['events'] = old['events']
        else:
            kept = []
            for e in sorted((clean_event(e) for e in events if isinstance(e, dict) and e.get('id')), key=snowflake_key):
                if e['id'] in hidden:
                    kept.append(only_what_approval_needs(e, 'hidden'))
                elif server['id'] in ask and approved.get(e['id']) != version_of(e):
                    kept.append(only_what_approval_needs(e, 'pending'))
                else:
                    kept.append(e)
            server['events'] = kept
    return out


def read_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def main(argv):
    if len(argv) not in (2, 3):
        print('usage: fetch_discord.py out_folder [previous_folder]', file=sys.stderr)
        return 2
    folder = argv[1]
    os.makedirs(folder, exist_ok=True)
    previous = read_json(os.path.join(argv[2], 'discord.json')) if len(argv) == 3 else None
    dispatched = os.environ.get('DISCORD_DISPATCHED', '').strip().lower() == 'true'
    if not dispatched and previous is not None:
        # Started by hand or by a change of this file, not by the sheet: no list of shown servers came
        # with it, so the last result stays as it is (it would otherwise lose every shown server's events).
        result = previous
        print('Discord: not started by the sheet; the last result stays.')
    else:
        result = fetch(os.environ.get('DISCORD_BOT_TOKEN', '').strip(),
                       allowed=allowed_servers(os.environ.get('DISCORD_SERVERS', '')),
                       ask=allowed_servers(os.environ.get('DISCORD_ASK', '')),
                       hidden=allowed_servers(os.environ.get('DISCORD_HIDDEN', '')),
                       approved=approved_versions(os.environ.get('DISCORD_APPROVED', '')),
                       previous=previous)
        if result['status'].startswith('error') and previous is not None:
            print('Discord: ' + result['status'] + ' — the last result stays.')
            result = previous
    with open(os.path.join(folder, 'discord.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
        f.write('\n')
    events = sum(len(s.get('events') or []) for s in result.get('servers') or [])
    print('Discord: ' + str(result.get('status')) + ', ' + str(len(result.get('servers') or [])) + ' servers, ' + str(events) + ' events.')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

# Error Code Reference

## Error SKU-4471

Error SKU-4471 indicates that the warehouse inventory sync failed for a
specific item. The usual cause is a stale product record in the upstream
catalogue. Resolution: re-run the catalogue sync job, then retry the order.

## Error SKU-8830

Error SKU-8830 means the item was found in the catalogue but has no assigned
warehouse location. Resolution: assign a bin location in the warehouse admin
tool, then retry.

## Error SKU-2205

Error SKU-2205 indicates a negative stock level was detected during
reconciliation. Resolution: freeze the item, run a manual stock count, then
correct the ledger.

## Error PAY-9902

Error PAY-9902 means the payment gateway rejected the transaction. This is
almost always an expired card or insufficient funds; it is not a system
fault. Resolution: ask the customer to update their payment method.

## Error PAY-1180

Error PAY-1180 indicates a timeout communicating with the payment provider.
Transactions in this state must be reconciled manually before any retry, or
the customer may be charged twice. Resolution: check the provider dashboard
before taking any action.

## Error PAY-3340

Error PAY-3340 means the payment provider returned a duplicate transaction
identifier. Resolution: do not retry; investigate whether the original
transaction succeeded.

## Error AUTH-5512

Error AUTH-5512 indicates the session token expired mid-request. Resolution:
the client should refresh the token and replay the request.

## Error AUTH-7701

Error AUTH-7701 means the API key was revoked. Resolution: issue a new key
through the developer portal; revoked keys are never reinstated.
